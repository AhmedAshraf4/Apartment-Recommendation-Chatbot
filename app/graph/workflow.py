import re

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.graph.state import ChatState
from app.services.detect_action import detect_action
from app.services.extract_contact_updates import extract_contact_updates_llm
from app.services.route_helpers import classify_routing_hint_llm
from app.services.followup_scope import classify_followup_scope
from app.services.lead_prepare import (
    build_success_reply,
    build_missing_reply,
    get_missing_fields,
)
from app.services.llm_chatbot import (
    extract_meta,
    search_apartments,
    search_reply_stream_to_writer,
    company_info_stream_to_writer,
    general_chat_stream_to_writer,
    apartment_followup_stream_to_writer,
    shown_apartments_followup_stream_to_writer,
    fallback_chat_stream_to_writer,
    get_apartment_by_exact_id,
)
from app.services.email_gen import send_email
from app.services.reference_helpers import (
    build_apartment_reference_map,
    get_apartment_by_id,
    resolve_apartment_reference,
)


def query_mentions_sorting(user_query: str) -> bool:
    query = str(user_query or "").lower()
    return any(
        token in query
        for token in [
            "sort",
            "sorted",
            "first",
            "ascending",
            "descending",
            "low to high",
            "high to low",
            "lowest to highest",
            "highest to lowest",
            "cheapest first",
            "largest first",
            "smallest first",
        ]
    )


def merge_search_filters(previous_filters: dict, new_filters: dict, user_query: str) -> dict:
    previous_filters = dict(previous_filters or {})
    new_filters = dict(new_filters or {})

    merged = dict(previous_filters)

    for key in [
        "title",
        "city",
        "min_bedrooms",
        "max_bedrooms",
        "min_bathrooms",
        "max_bathrooms",
        "min_price",
        "max_price",
        "view",
    ]:
        if new_filters.get(key) is not None:
            merged[key] = new_filters.get(key)

    if query_mentions_sorting(user_query):
        merged["sort_by"] = new_filters.get("sort_by", merged.get("sort_by", "price"))
        merged["sort_order"] = new_filters.get("sort_order", merged.get("sort_order", "asc"))
    else:
        merged["sort_by"] = previous_filters.get("sort_by", new_filters.get("sort_by", "price"))
        merged["sort_order"] = previous_filters.get("sort_order", new_filters.get("sort_order", "asc"))

    return merged


def extract_direct_apartment_id(user_message: str) -> str | None:
    query = str(user_message or "").strip().lower()
    match = re.search(r"\b(?:ap|th|dp|ph)\d{3}\b", query)
    if match:
        return match.group(0)
    return None


def build_focus_update(apartment: dict | None) -> dict:
    if not apartment:
        return {}

    apartment_id = apartment.get("apartment_id")
    if not apartment_id:
        return {}

    return {
        "selected_apartment_id": apartment_id,
        "selected_apartment": apartment,
    }


def append_interest_hint(reply: str) -> str:
    reply = str(reply or "").strip()
    hint = 'If you like a specific apartment, just say "I want this."'

    if not reply:
        return hint

    if hint.lower() in reply.lower():
        return reply

    return f"{reply}\n\n{hint}"


def resolve_single_apartment_from_shown_list(user_query: str, apartments: list[dict]) -> dict | None:
    query = str(user_query or "").strip().lower()
    apartments = apartments or []

    if not apartments:
        return None

    def safe_float(value, default):
        try:
            return float(value)
        except Exception:
            return default

    if any(term in query for term in ["largest area", "largest", "biggest", "more space", "most spacious"]):
        return max(apartments, key=lambda x: safe_float(x.get("area_sqm"), float("-inf")))

    if any(term in query for term in ["smallest", "least spacious"]):
        return min(apartments, key=lambda x: safe_float(x.get("area_sqm"), float("inf")))

    if any(term in query for term in ["most expensive", "highest price", "priciest"]):
        return max(apartments, key=lambda x: safe_float(x.get("price"), float("-inf")))

    if any(term in query for term in ["cheapest", "lowest price", "least expensive"]):
        return min(apartments, key=lambda x: safe_float(x.get("price"), float("inf")))

    if any(term in query for term in ["most bedrooms", "most number of bedrooms", "highest number of bedrooms", "more bedrooms"]):
        return max(apartments, key=lambda x: safe_float(x.get("bedrooms"), float("-inf")))

    if any(term in query for term in ["most bathrooms", "highest number of bathrooms", "more bathrooms"]):
        return max(apartments, key=lambda x: safe_float(x.get("bathrooms"), float("-inf")))

    return None


def llm_followup_scope(user_message: str, state) -> str:
    result = classify_followup_scope(user_message, state)
    return result.get("scope", "none")


def resolve_action_apartment(state):
    action_result = state.get("action_result", {}) or {}
    reference_type = action_result.get("reference_type", "none")
    reference_value = action_result.get("reference_value")

    user_query = state.get("user_query", "")
    lowered_query = str(user_query or "").strip().lower()

    last_shown = state.get("last_shown_apartments", []) or []
    ref_map = state.get("apartment_reference_map", {}) or {}

    # Direct apartment ID in current turn always wins
    direct_apartment_id = extract_direct_apartment_id(user_query)
    if direct_apartment_id:
        apartment = get_apartment_by_exact_id(direct_apartment_id)
        return direct_apartment_id, apartment

    apartment_id = None

    # Explicit ordinal in current turn
    for alias, mapped_id in ref_map.items():
        if alias in lowered_query:
            apartment_id = mapped_id
            break

    # Structured action result / helper fallback
    if apartment_id is None:
        if reference_type == "id" and reference_value:
            apartment_id = reference_value
        elif reference_type == "ordinal" and reference_value:
            apartment_id = ref_map.get(str(reference_value).strip().lower())
        elif reference_type == "selected":
            apartment_id = state.get("selected_apartment_id")
        else:
            apartment_id = resolve_apartment_reference(user_query, state)

    # Current shown apartments first
    apartment = get_apartment_by_id(last_shown, apartment_id)

    # Exact global lookup if action explicitly references ID
    if apartment is None and reference_type == "id" and apartment_id:
        apartment = get_apartment_by_exact_id(apartment_id)

    # Fallback to selected apartment only for non-ID conversational references
    if apartment is None and reference_type != "id" and state.get("selected_apartment"):
        selected = state.get("selected_apartment")
        selected_id = selected.get("apartment_id")

        if apartment_id:
            if str(selected_id or "").strip().lower() == str(apartment_id).strip().lower():
                apartment = selected
        else:
            apartment = selected
            apartment_id = selected_id

    return apartment_id, apartment


def is_submit_request(user_message: str) -> bool:
    query = str(user_message or "").strip().lower()
    return any(
        phrase in query
        for phrase in [
            "proceed",
            "continue",
            "go ahead",
            "submit",
            "send it",
            "send my request",
            "use this apartment",
            "contact them",
            "follow up",
            "request callback",
        ]
    )


def looks_like_explicit_search_request(user_message: str, state) -> bool:
    query = str(user_message or "").strip().lower()

    list_followup_terms = [
        "above",
        "these",
        "those",
        "options",
        "compare",
        "which one",
        "which of the above",
        "which of these",
        "among these",
        "list the apartments above",
        "list the above",
        "largest area",
        "largest one",
        "smallest one",
        "most expensive",
        "cheapest",
    ]
    if any(term in query for term in list_followup_terms):
        return False

    search_phrases = [
        "i want",
        "iwant",
        "i need",
        "show me",
        "give me",
        "find me",
        "looking for",
        "search for",
    ]

    property_terms = [
        "apartment",
        "appartment",
        "studio",
        "townhouse",
        "townhouses",
        "penthouse",
        "duplex",
        "unit",
        "property",
        "properties",
    ]

    location_terms = [
        "zayed",
        "sheikh zayed",
        "october",
        "6 october",
        "new cairo",
        "tagamoa",
        "fifth settlement",
        "rehab",
        "madinaty",
        "north coast",
        "ain sokhna",
        "new capital",
    ]

    filter_terms = [
        "bedroom",
        "bathroom",
        "view",
        "price",
        "under",
        "over",
        "budget",
        "sqm",
        "garden",
        "pool",
        "gym",
        "parking",
    ]

    if any(phrase in query for phrase in search_phrases):
        if any(term in query for term in property_terms + location_terms + filter_terms):
            return True

    hint = classify_routing_hint_llm(user_message, state)
    return hint.get("route_hint") == "search"


def looks_like_selected_apartment_followup(user_message: str, state) -> bool:
    query = str(user_message or "").strip().lower()

    explicit_reference_terms = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "option 1",
        "option 2",
        "option 3",
        "option 4",
        "option 5",
        "the first one",
        "the second one",
        "the third one",
    ]
    if any(term in query for term in explicit_reference_terms):
        return False

    pronoun_terms = [
        "it",
        "its",
        "this one",
        "that one",
        "this apartment",
        "that apartment",
        "this property",
        "that property",
    ]

    detail_terms = [
        "does",
        "have",
        "price",
        "bedroom",
        "bathroom",
        "area",
        "view",
        "amenities",
        "pool",
        "gym",
        "parking",
        "garden",
        "details",
        "info",
        "more",
        "tell me",
        "what about",
        "better",
        "compare",
    ]

    if (
        state.get("selected_apartment_id") is not None
        and any(term in query for term in pronoun_terms)
        and any(term in query for term in detail_terms)
    ):
        return True

    hint = classify_routing_hint_llm(user_message, state)
    return hint.get("route_hint") == "selected_followup"


def looks_like_selected_apartment_selection(user_message: str, state) -> bool:
    query = str(user_message or "").strip().lower()

    if not state.get("selected_apartment_id"):
        return False

    selection_terms = [
        "i want it",
        "yes i want it",
        "want it",
        "i like it",
        "i'll take it",
        "i will take it",
        "take it",
        "use it",
        "use this one",
        "use that one",
        "i want this one",
        "i want that one",
        "this one",
        "that one",
        "i want this apartment",
        "i want that apartment",
        "i want this",
        "iwant this",
        "iwant it",
    ]

    return any(term in query for term in selection_terms)


def looks_like_explicit_apartment_selection(user_message: str, state) -> bool:
    query = str(user_message or "").strip().lower()

    strong_selection_phrases = [
        "i want",
        "iwant",
        "i like",
        "i choose",
        "choose",
        "select",
        "take",
        "i'll take",
        "i will take",
        "use",
        "go with",
    ]

    plain_reference_selection_terms = [
        "first one",
        "second one",
        "third one",
        "fourth one",
        "fifth one",
        "the first one",
        "the second one",
        "the third one",
        "the fourth one",
        "the fifth one",
        "first option",
        "second option",
        "third option",
        "fourth option",
        "fifth option",
        "the first option",
        "the second option",
        "the third option",
        "the fourth option",
        "the fifth option",
        "option 1",
        "option 2",
        "option 3",
        "option 4",
        "option 5",
    ]

    shown = state.get("last_shown_apartments") or []

    has_id_ref = False
    for apartment in shown:
        apartment_id = str(apartment.get("apartment_id", "")).strip().lower()
        if apartment_id and apartment_id in query:
            has_id_ref = True
            break

    if has_id_ref and any(p in query for p in strong_selection_phrases):
        return True

    if any(ref in query for ref in plain_reference_selection_terms) and any(
        p in query for p in strong_selection_phrases
    ):
        return True

    if query in plain_reference_selection_terms:
        return True

    return False


def looks_like_confirmation_to_proceed(user_message: str, state) -> bool:
    query = str(user_message or "").strip().lower()
    if query not in {"yes", "yeah", "yep", "sure", "ok", "okay"}:
        return False

    history = state.get("chat_history") or []
    if not history:
        return False

    last_assistant = None
    for item in reversed(history):
        if item.get("role") == "assistant":
            last_assistant = str(item.get("content", "")).lower()
            break

    if not last_assistant:
        return False

    proceed_prompts = [
        "proceed",
        "continue with this apartment",
        "send me your name, email, and phone number",
        "ready to move forward",
        "tell me to proceed",
    ]

    return any(term in last_assistant for term in proceed_prompts)


def looks_like_shown_list_followup(user_message: str, state) -> bool:
    query = str(user_message or "").strip().lower()
    shown = state.get("last_shown_apartments") or []

    if not shown:
        return False

    terms = [
        "above",
        "these",
        "those",
        "options",
        "compare",
        "compare the options",
        "list the apartments above",
        "list the above",
        "which one",
        "which of the above",
        "which of these",
        "among these",
        "most expensive",
        "highest price",
        "priciest",
        "cheapest",
        "lowest price",
        "least expensive",
        "largest",
        "largest area",
        "one with largest area",
        "one with the largest area",
        "biggest",
        "smallest",
        "better",
        "best",
        "which options",
        "which apartments",
        "most bedrooms",
        "most number of bedrooms",
        "highest number of bedrooms",
        "more bedrooms",
        "most bathrooms",
        "highest number of bathrooms",
        "more bathrooms",
    ]

    return any(term in query for term in terms)


def looks_like_numeric_option_selection(user_message: str, state) -> bool:
    query = str(user_message or "").strip().lower()
    shown = state.get("last_shown_apartments") or []
    if not shown:
        return False
    return query in {"1", "2", "3", "4", "5"}


def action_node(state):
    user_message = state["user_query"]
    lowered = str(user_message or "").strip().lower()

    direct_apartment_id = extract_direct_apartment_id(user_message)

    # 0) Direct apartment ID always wins
    if direct_apartment_id:
        selection_phrases = [
            "i want",
            "iwant",
            "select",
            "choose",
            "take",
            "use",
            "i like",
            "go with",
        ]

        if any(phrase in lowered for phrase in selection_phrases):
            return {
                "action_result": {
                    "action": "select_apartment",
                    "search_mode": "none",
                    "reference_type": "id",
                    "reference_value": direct_apartment_id,
                    "field_updates": {},
                    "source": "rule_based_direct_id_selection",
                },
                "action": "select_apartment",
            }

        return {
            "action_result": {
                "action": "get_apartment_details",
                "search_mode": "none",
                "reference_type": "id",
                "reference_value": direct_apartment_id,
                "field_updates": {},
                "source": "rule_based_direct_id_details",
            },
            "action": "get_apartment_details",
        }

    # 1) Numeric shortlist selection like "1"
    if looks_like_numeric_option_selection(user_message, state):
        return {
            "action_result": {
                "action": "select_apartment",
                "search_mode": "none",
                "reference_type": "ordinal",
                "reference_value": lowered,
                "field_updates": {},
                "source": "rule_based_numeric_selection",
            },
            "action": "select_apartment",
        }

    # 2) Explicit ordinal / shown-item selection MUST beat list analysis
    if looks_like_explicit_apartment_selection(user_message, state):
        action_result = detect_action(user_message, state)

        reference_type = action_result.get("reference_type", "none")
        reference_value = action_result.get("reference_value")

        if reference_type == "none":
            if "first" in lowered or "option 1" in lowered:
                reference_type = "ordinal"
                reference_value = "first"
            elif "second" in lowered or "option 2" in lowered:
                reference_type = "ordinal"
                reference_value = "second"
            elif "third" in lowered or "option 3" in lowered:
                reference_type = "ordinal"
                reference_value = "third"
            elif "fourth" in lowered or "option 4" in lowered:
                reference_type = "ordinal"
                reference_value = "fourth"
            elif "fifth" in lowered or "option 5" in lowered:
                reference_type = "ordinal"
                reference_value = "fifth"

        return {
            "action_result": {
                "action": "select_apartment",
                "search_mode": "none",
                "reference_type": reference_type,
                "reference_value": reference_value,
                "field_updates": {},
                "source": "rule_based_explicit_selection",
            },
            "action": "select_apartment",
        }

    # 3) Selected apartment conversational selection like "i want this"
    if looks_like_selected_apartment_selection(user_message, state):
        return {
            "action_result": {
                "action": "select_apartment",
                "search_mode": "none",
                "reference_type": "selected",
                "reference_value": state.get("selected_apartment_id"),
                "field_updates": {},
                "source": "rule_based_selected_selection",
            },
            "action": "select_apartment",
        }

    # 4) Submit / proceed MUST come before selected-apartment follow-up
    if looks_like_confirmation_to_proceed(user_message, state) or is_submit_request(user_message):
        return {
            "action_result": {
                "action": "submit_lead",
                "search_mode": "none",
                "reference_type": "selected" if state.get("selected_apartment_id") else "none",
                "reference_value": state.get("selected_apartment_id"),
                "field_updates": {},
                "source": "rule_based_submit",
            },
            "action": "submit_lead",
        }

    # 5) Selected apartment follow-up like "does it have a pool?"
    if looks_like_selected_apartment_followup(user_message, state):
        return {
            "action_result": {
                "action": "get_apartment_details",
                "search_mode": "none",
                "reference_type": "selected",
                "reference_value": state.get("selected_apartment_id"),
                "field_updates": {},
                "source": "rule_based_selected_followup",
            },
            "action": "get_apartment_details",
        }

    # 6) LLM rescue for ambiguous short follow-ups / missed cases
    scope = llm_followup_scope(user_message, state)

    if scope == "shown_list":
        return {
            "action_result": {
                "action": "analyze_shown_apartments",
                "search_mode": "none",
                "reference_type": "none",
                "reference_value": None,
                "field_updates": {},
                "source": "llm_followup_scope_shown_list",
            },
            "action": "analyze_shown_apartments",
        }

    if scope == "selected_apartment" and state.get("selected_apartment_id"):
        return {
            "action_result": {
                "action": "get_apartment_details",
                "search_mode": "none",
                "reference_type": "selected",
                "reference_value": state.get("selected_apartment_id"),
                "field_updates": {},
                "source": "llm_followup_scope_selected",
            },
            "action": "get_apartment_details",
        }

    if scope == "new_search":
        return {
            "action_result": {
                "action": "search",
                "search_mode": "refine" if state.get("last_search_filters") else "new",
                "reference_type": "none",
                "reference_value": None,
                "field_updates": {},
                "source": "llm_followup_scope_search",
            },
            "action": "search",
        }

    # 7) Shown-list follow-up only after explicit selection checks
    if looks_like_shown_list_followup(user_message, state):
        return {
            "action_result": {
                "action": "analyze_shown_apartments",
                "search_mode": "none",
                "reference_type": "none",
                "reference_value": None,
                "field_updates": {},
                "source": "rule_based_shown_list_followup",
            },
            "action": "analyze_shown_apartments",
        }

    # 8) Explicit new search
    if looks_like_explicit_search_request(user_message, state):
        return {
            "action_result": {
                "action": "search",
                "search_mode": "refine" if state.get("last_search_filters") else "new",
                "reference_type": "none",
                "reference_value": None,
                "field_updates": {},
                "source": "rule_based_explicit_search",
            },
            "action": "search",
        }

    # 9) Main LLM planner
    action_result = detect_action(user_message, state)
    action = action_result.get("action", "unsupported")

    # 10) Contact extraction only after main routing
    if action in {"unsupported", "update_lead_data", "submit_lead"}:
        llm_contact_updates = extract_contact_updates_llm(user_message, state)
        if llm_contact_updates:
            return {
                "action_result": {
                    "action": "update_lead_data",
                    "search_mode": "none",
                    "reference_type": "none",
                    "reference_value": None,
                    "field_updates": llm_contact_updates,
                    "source": "llm_contact_update",
                },
                "action": "update_lead_data",
            }

    # 11) Final fallback chat
    if action == "unsupported":
        action = "fallback_chat"
        action_result["action"] = "fallback_chat"

    return {
        "action_result": action_result,
        "action": action,
    }


def search_node(state):
    user_message = state["user_query"]
    action_result = state.get("action_result", {}) or {}
    search_mode = action_result.get("search_mode", "new")

    extracted_filters = extract_meta(user_message)

    if search_mode == "refine" and state.get("last_search_filters"):
        filters = merge_search_filters(state.get("last_search_filters", {}), extracted_filters, user_message)
    else:
        filters = extracted_filters

    matches = search_apartments(user_message, filters, 15)
    reply = search_reply_stream_to_writer(user_message, filters, matches)
    reply = append_interest_hint(reply)
    reference_map = build_apartment_reference_map(matches)

    return {
        "filters": filters,
        "last_search_filters": filters,
        "matches": matches,
        "last_shown_apartments": matches,
        "apartment_reference_map": reference_map,
        "reply": reply,
        "stream_text": reply,
    }


@traceable(name="company_info_node")
def company_info_node(state):
    user_message = state["user_query"]
    reply = company_info_stream_to_writer(user_message)

    return {
        "reply": reply,
        "stream_text": reply,
    }


def apartment_details_node(state):
    apartment_id, apartment = resolve_action_apartment(state)

    if apartment is None:
        requested = apartment_id or "that apartment"
        reply = (
            f"I could not find {requested} in the current session or by exact apartment ID. "
            f"Please check the ID or ask for apartments again."
        )
        return {
            "reply": reply,
            "stream_text": reply,
        }

    user_query = state.get("user_query", "")
    lowered = str(user_query or "").lower()

    reference_label = "this apartment"
    if "first" in lowered:
        reference_label = "the first option"
    elif "second" in lowered:
        reference_label = "the second option"
    elif "third" in lowered:
        reference_label = "the third option"
    elif "fourth" in lowered:
        reference_label = "the fourth option"
    elif "fifth" in lowered:
        reference_label = "the fifth option"
    elif apartment_id:
        reference_label = f"apartment {apartment_id}"

    reply = apartment_followup_stream_to_writer(
        user_query=user_query,
        apartment=apartment,
        reference_label=reference_label,
    )
    reply = append_interest_hint(reply)

    return {
        **build_focus_update(apartment),
        "reply": reply,
        "stream_text": reply,
    }


def shown_apartments_analysis_node(state):
    apartments = state.get("last_shown_apartments") or []

    if not apartments:
        reply = "I do not have any currently shown apartments to analyze. Please ask for apartments first."
        return {
            "reply": reply,
            "stream_text": reply,
        }

    focused_apartment = resolve_single_apartment_from_shown_list(
        state.get("user_query", ""),
        apartments,
    )

    reply = shown_apartments_followup_stream_to_writer(
        state.get("user_query", ""),
        apartments,
    )
    reply = append_interest_hint(reply)

    return {
        **build_focus_update(focused_apartment),
        "reply": reply,
        "stream_text": reply,
    }


def apartment_selected_node(state):
    apartment_id, apartment = resolve_action_apartment(state)

    if apartment is None:
        requested = apartment_id or "that apartment"
        reply = (
            f"I could not find {requested} in the current results or by exact apartment ID. "
            f"Please check the ID and try again."
        )
        return {
            "reply": reply,
            "stream_text": reply,
        }

    title = apartment.get("title", "property")
    city = apartment.get("city", "N/A")
    area = apartment.get("area", "N/A")
    price = apartment.get("price", "N/A")

    reply = (
        f"Got it — you selected apartment {apartment_id}.\n\n"
        f"It is a {title} in {city} - {area} priced at {price} EGP.\n\n"
        f"You can now send me your name, email, and phone number to continue, "
        f"or tell me to proceed with this apartment."
    )

    lead_data = dict(state.get("lead_data", {}) or {})
    lead_data["apartment_id"] = apartment_id

    return {
        **build_focus_update(apartment),
        "lead_data": lead_data,
        "reply": reply,
        "stream_text": reply,
    }


def update_lead_data_node(state):
    action_result = state.get("action_result", {}) or {}
    field_updates = dict(action_result.get("field_updates", {}) or {})

    if not field_updates:
        field_updates = extract_contact_updates_llm(state.get("user_query", ""), state)

    current_lead = dict(state.get("lead_data", {}) or {})
    user_profile = dict(state.get("user_profile", {}) or {})

    # Build a hydrated lead from remembered profile first
    hydrated_lead = {
        "name": current_lead.get("name") or user_profile.get("name"),
        "email": current_lead.get("email") or user_profile.get("email"),
        "phone": current_lead.get("phone") or user_profile.get("phone"),
        "preferred_contact_time": current_lead.get("preferred_contact_time") or user_profile.get("preferred_contact_time"),
        "apartment_id": current_lead.get("apartment_id") or state.get("selected_apartment_id") or user_profile.get("apartment_id"),
    }

    # Apply new updates on top
    merged_lead = {**hydrated_lead, **field_updates}
    updated_profile = {**user_profile, **field_updates}

    changed_fields = ", ".join(field_updates.keys()) if field_updates else "details"
    missing_fields = get_missing_fields(merged_lead)

    if field_updates:
        if missing_fields:
            reply = (
                f"I updated your {changed_fields}. "
                f"I still need: {', '.join(missing_fields)}."
            )
        else:
            reply = (
                f"I updated your {changed_fields}. "
                f"Your details look complete now. Tell me to proceed when you’re ready."
            )
    else:
        reply = (
            "I still could not detect any new contact details. "
            "Please send your name, email, phone number, or preferred contact time."
        )

    return {
        "lead_data": merged_lead,
        "user_profile": updated_profile,
        "missing_fields": missing_fields,
        "reply": reply,
        "stream_text": reply,
    }


def submit_lead_prep_node(state):
    user_profile = state.get("user_profile", {}) or {}
    lead_data = dict(state.get("lead_data", {}) or {})

    hydrated_lead = {
        "name": lead_data.get("name") or user_profile.get("name"),
        "email": lead_data.get("email") or user_profile.get("email"),
        "phone": lead_data.get("phone") or user_profile.get("phone"),
        "preferred_contact_time": lead_data.get("preferred_contact_time") or user_profile.get("preferred_contact_time"),
        "apartment_id": lead_data.get("apartment_id") or state.get("selected_apartment_id"),
    }

    missing_fields = get_missing_fields(hydrated_lead)

    return {
        "lead_data": hydrated_lead,
        "missing_fields": missing_fields,
    }


def missing_lead_info_node(state):
    lead_data = state.get("lead_data", {})
    missing_fields = state.get("missing_fields", [])
    reply = build_missing_reply(lead_data, missing_fields)

    return {
        "reply": reply,
        "stream_text": reply,
    }


def send_lead_node(state):
    lead_data = state.get("lead_data", {})
    matches = state.get("last_shown_apartments") or state.get("matches", [])

    apartment_id = str(lead_data.get("apartment_id", "")).strip().lower()
    selected_apartment = None

    for match in matches:
        if str(match.get("apartment_id", "")).strip().lower() == apartment_id:
            selected_apartment = match
            break

    if selected_apartment is None and state.get("selected_apartment"):
        current_selected = state.get("selected_apartment")
        if str(current_selected.get("apartment_id", "")).strip().lower() == apartment_id:
            selected_apartment = current_selected

    if selected_apartment is None:
        reply = (
            f"I have all the lead details, but I could not find apartment {apartment_id} "
            f"in the current session results. Please search for it again, then resend your request."
        )
        return {
            "reply": reply,
            "stream_text": reply,
        }

    result = send_email(selected_apartment, lead_data)

    if result.get("success"):
        reply = build_success_reply(lead_data)

        remembered_profile = dict(state.get("user_profile", {}) or {})
        remembered_profile.update(
            {
                "name": lead_data.get("name") or remembered_profile.get("name"),
                "email": lead_data.get("email") or remembered_profile.get("email"),
                "phone": lead_data.get("phone") or remembered_profile.get("phone"),
                "preferred_contact_time": lead_data.get("preferred_contact_time") or remembered_profile.get(
                    "preferred_contact_time"),
            }
        )

        return {
            "reply": reply,
            "stream_text": reply,
            "lead_data": {},
            "missing_fields": [],
            "user_profile": remembered_profile,
        }


def general_chat_node(state):
    user_message = state["user_query"]
    reply = general_chat_stream_to_writer(user_message, state)

    return {
        "reply": reply,
        "stream_text": reply,
    }


def fallback_chat_node(state):
    user_message = state["user_query"]
    reply = fallback_chat_stream_to_writer(user_message, state)

    return {
        "reply": reply,
        "stream_text": reply,
    }


def unsupported_node(state):
    reply = (
        "I can help with apartment search, apartment details, selecting an apartment, "
        "updating your contact details, lead requests, or Dorra company information only.\n"
        "Try asking for a property, asking about one of the shown options, "
        "sharing your contact details, or asking about Dorra.\n"
        "Or you can contact one of our sales team via Hotline: 16077 or Email: info@dorra.com"
    )
    return {
        "reply": reply,
        "stream_text": reply,
    }


def build_chat_graph():
    graph = StateGraph(ChatState)

    graph.add_node("detect_action", action_node)
    graph.add_node("search_and_recommend", search_node)
    graph.add_node("company_info", company_info_node)
    graph.add_node("apartment_details", apartment_details_node)
    graph.add_node("shown_apartments_analysis", shown_apartments_analysis_node)
    graph.add_node("apartment_selected", apartment_selected_node)
    graph.add_node("update_lead_data", update_lead_data_node)
    graph.add_node("submit_lead_prep", submit_lead_prep_node)
    graph.add_node("lead_reply", missing_lead_info_node)
    graph.add_node("send_lead", send_lead_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("fallback_chat", fallback_chat_node)
    graph.add_node("unsupported", unsupported_node)

    graph.add_edge(START, "detect_action")

    graph.add_conditional_edges(
        "detect_action",
        lambda state: state.get("action", "unsupported"),
        {
            "search": "search_and_recommend",
            "company_info": "company_info",
            "get_apartment_details": "apartment_details",
            "select_apartment": "apartment_selected",
            "analyze_shown_apartments": "shown_apartments_analysis",
            "update_lead_data": "update_lead_data",
            "submit_lead": "submit_lead_prep",
            "general_chat": "general_chat",
            "fallback_chat": "fallback_chat",
            "unsupported": "unsupported",
        },
    )

    graph.add_edge("search_and_recommend", END)
    graph.add_edge("company_info", END)
    graph.add_edge("apartment_details", END)
    graph.add_edge("shown_apartments_analysis", END)
    graph.add_edge("apartment_selected", END)
    graph.add_edge("update_lead_data", END)
    graph.add_edge("general_chat", END)
    graph.add_edge("fallback_chat", END)
    graph.add_edge("unsupported", END)

    graph.add_conditional_edges(
        "submit_lead_prep",
        lambda state: "missing" if state.get("missing_fields", []) else "complete",
        {
            "missing": "lead_reply",
            "complete": "send_lead",
        },
    )

    graph.add_edge("lead_reply", END)
    graph.add_edge("send_lead", END)

    return graph.compile()


chat_graph = build_chat_graph()