import json
import re

from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.core.config import settings
from app.graph.state import ChatState
from app.services.agent_schedule_db import (
    check_booking_time_availability,
    reserve_booking_time_slot,
)
from app.services.email_gen import send_email
from app.services.extract_contact_updates import extract_contact_updates_llm
from app.services.lead_prepare import (
    format_iso_for_display,
    get_missing_fields,
)
from app.services.llm_action_planner import plan_action_llm
from app.services.llm_chatbot import (
    apartment_followup_stream_to_writer,
    company_info_stream_to_writer,
    extract_meta,
    fallback_chat_stream_to_writer,
    general_chat_stream_to_writer,
    get_apartment_by_exact_id,
    search_apartments,
    search_reply_stream_to_writer,
    shown_apartments_followup_stream_to_writer,
)
from app.services.reference_helpers import (
    build_apartment_reference_map,
    get_apartment_by_id,
    resolve_apartment_reference,
)



import json
from langchain_openai import ChatOpenAI

from app.core.config import settings


def _safe_json(text: str):
    text = str(text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None

    return None



def classify_id_intent_llm(user_message: str, apartment_id: str, state: dict) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    context = {
        "latest_user_message": user_message,
        "apartment_id_in_message": apartment_id,
        "recent_history": (state.get("chat_history") or [])[-6:],
        "selected_apartment": {
            "apartment_id": (state.get("selected_apartment") or {}).get("apartment_id"),
            "title": (state.get("selected_apartment") or {}).get("title"),
        },
        "shown_apartments": [
            {
                "order": i + 1,
                "apartment_id": apt.get("apartment_id"),
                "title": apt.get("title"),
            }
            for i, apt in enumerate((state.get("last_shown_apartments") or [])[:10])
        ],
        "pending_confirmation": state.get("pending_confirmation") or {},
        "pending_compare": state.get("pending_compare") or {},
    }

    prompt = f"""
    You classify what the user intends when they mention an apartment ID.

    Return JSON only.
    Do not explain anything.
    Do not output markdown.

    Allowed actions:
    - "get_apartment_details"
    - "compare_apartments"
    - "select_apartment"
    - "submit_lead"

    Default behavior:
    If the message is only a bare apartment ID with no clear intent, choose "get_apartment_details".

    Priority order:
    1. If the user is clearly CHOOSING an apartment by explicit ID, choose "select_apartment".
    2. If the user is clearly COMPARING apartments, choose "compare_apartments".
    3. If the user is clearly asking to proceed/contact/book for that apartment, choose "submit_lead".
    4. Otherwise choose "get_apartment_details".

    Very important:
    Explicit apartment choice by ID is STRONGER than prior submit/lead context.
    If the latest message clearly chooses a different apartment by ID, do NOT keep the old apartment and do NOT choose "submit_lead".

    Examples that MUST be "select_apartment":
    - "i want ap023"
    - "i choose ap023"
    - "go with ap023"
    - "use ap023"
    - "i want apartment ap023"
    - "i'll take ap023"

    Examples that may be "submit_lead":
    - "submit ap023"
    - "proceed with ap023"
    - "book ap023"
    - "contact them for ap023"

    Examples that should be "get_apartment_details":
    - "ap023"
    - "details about ap023"
    - "tell me about ap023"

    Examples that should be "compare_apartments":
    - "compare ap023 with ap025"
    - "is ap023 better than ap025"

    Return exactly this schema:
    {{
      "action": "get_apartment_details",
      "confidence": 0.82,
      "reason": "bare apartment id with no strong competing intent"
    }}

    Context:
    {json.dumps(context, ensure_ascii=False, indent=2)}
    """.strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = _safe_json(raw) or {}

    action = str(parsed.get("action") or "").strip()
    if action not in {
        "get_apartment_details",
        "compare_apartments",
        "select_apartment",
        "submit_lead",
    }:
        action = "get_apartment_details"

    return {
        "action": action,
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason"),
    }


def validate_booking_availability_for_lead_time(state: dict, lead_snapshot: dict) -> dict:
    lead_snapshot = dict(lead_snapshot or {})
    apartment_id = str(lead_snapshot.get("apartment_id") or "").strip().lower()
    user_email = str(lead_snapshot.get("email") or "").strip().lower()
    requested_contact_at_iso = str(lead_snapshot.get("preferred_contact_time_iso") or "").strip()

    if not apartment_id or not user_email or not requested_contact_at_iso:
        return {"success": True, "skipped": True}

    selected_apartment = None

    for apt in (state.get("last_shown_apartments") or []):
        if str(apt.get("apartment_id") or "").strip().lower() == apartment_id:
            selected_apartment = apt
            break

    if selected_apartment is None and state.get("selected_apartment"):
        current_selected = state.get("selected_apartment") or {}
        if str(current_selected.get("apartment_id") or "").strip().lower() == apartment_id:
            selected_apartment = current_selected

    if selected_apartment is None:
        selected_apartment = get_apartment_by_exact_id(apartment_id)

    if selected_apartment is None:
        return {"success": True, "skipped": True}

    agent_email = str(selected_apartment.get("agent_email") or "").strip().lower()
    if not agent_email:
        return {"success": True, "skipped": True}

    return check_booking_time_availability(
        agent_email=agent_email,
        user_email=user_email,
        requested_contact_at_iso=requested_contact_at_iso,
        apartment_id=apartment_id,
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


def _stream_llm_text(prompt: str, temperature: float = 0.25) -> str:
    writer = get_stream_writer()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )

    collected = []
    for chunk in llm.stream(prompt):
        text = chunk.content or ""
        if not isinstance(text, str):
            text = str(text)
        if text:
            collected.append(text)
            writer(text)

    return "".join(collected).strip()


def _stream_suffix(base_reply: str, suffix: str) -> str:
    base_reply = str(base_reply or "").strip()
    suffix = str(suffix or "").strip()

    if not suffix:
        return base_reply

    if suffix.lower() in base_reply.lower():
        return base_reply

    writer = get_stream_writer()
    extra = f"\n\n{suffix}" if base_reply else suffix
    writer(extra)
    return f"{base_reply}{extra}" if base_reply else suffix


def stream_interest_hint(reply: str) -> str:
    return _stream_suffix(
        reply,
        'If you want this apartment, just say "I want this" or "I want it."',
    )


def normalize_planner_action(action: str) -> str:
    allowed_actions = {
        "search",
        "get_apartment_details",
        "select_apartment",
        "analyze_shown_apartments",
        "update_lead_data",
        "submit_lead",
        "company_info",
        "general_chat",
        "reply_direct",
        "fallback_chat",
        "unsupported",
    }
    action = str(action or "").strip()
    if action not in allowed_actions:
        return "fallback_chat"
    return action


def extract_direct_apartment_id(user_message: str) -> str | None:
    query = str(user_message or "").strip().lower()
    match = re.search(r"\b[a-z]{2,5}\d{3,6}\b", query)
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

    if any(
        term in query
        for term in [
            "most bedrooms",
            "most number of bedrooms",
            "highest number of bedrooms",
            "more bedrooms",
        ]
    ):
        return max(apartments, key=lambda x: safe_float(x.get("bedrooms"), float("-inf")))

    if any(
        term in query
        for term in [
            "most bathrooms",
            "highest number of bathrooms",
            "more bathrooms",
        ]
    ):
        return max(apartments, key=lambda x: safe_float(x.get("bathrooms"), float("-inf")))

    return None


def is_yes_message(text: str) -> bool:
    value = str(text or "").strip().lower()
    return re.search(r"\b(yes|yeah|yep|sure|ok|okay)\b", value) is not None


def is_no_message(text: str) -> bool:
    value = str(text or "").strip().lower()
    return re.search(r"\b(no|nope|nah)\b", value) is not None


def resolve_action_apartment(state):
    action_result = state.get("action_result", {}) or {}
    reference_type = action_result.get("reference_type", "none")
    reference_value = action_result.get("reference_value")

    user_query = state.get("user_query", "")
    lowered_query = str(user_query or "").strip().lower()

    last_shown = state.get("last_shown_apartments", []) or []
    ref_map = state.get("apartment_reference_map", {}) or {}

    direct_apartment_id = extract_direct_apartment_id(user_query)
    if direct_apartment_id:
        apartment = get_apartment_by_exact_id(direct_apartment_id)
        return direct_apartment_id, apartment

    apartment_id = None

    for alias, mapped_id in ref_map.items():
        if alias in lowered_query:
            apartment_id = mapped_id
            break

    if apartment_id is None:
        if reference_type == "id" and reference_value:
            apartment_id = reference_value
        elif reference_type == "ordinal" and reference_value:
            apartment_id = ref_map.get(str(reference_value).strip().lower())
        elif reference_type == "selected":
            apartment_id = state.get("selected_apartment_id")
        else:
            apartment_id = resolve_apartment_reference(user_query, state)

    apartment = get_apartment_by_id(last_shown, apartment_id)

    if apartment is None and reference_type == "id" and apartment_id:
        apartment = get_apartment_by_exact_id(apartment_id)

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


def lead_status_stream_to_writer(
    *,
    user_query: str,
    lead_data: dict,
    missing_fields: list[str],
    just_completed: bool,
    pending_confirmation: bool,
) -> str:
    safe_context = {
        "user_query": user_query,
        "lead_data": {
            "name": lead_data.get("name"),
            "email": lead_data.get("email"),
            "phone": lead_data.get("phone"),
            "preferred_contact_time": lead_data.get("preferred_contact_time"),
            "apartment_id": lead_data.get("apartment_id"),
        },
        "missing_fields": missing_fields,
        "just_completed": just_completed,
        "pending_confirmation": pending_confirmation,
    }

    prompt = f"""
You are Dorra's real-estate assistant.

Write one short natural user-facing reply.

Rules:
1. Be warm and concise.
2. Do not greet the user unless they greeted in this message.
3. Do not repeat that old details are still saved unless directly relevant.
4. If some details are still missing, say naturally what is still needed.
5. If all details are complete, tell the user to say "proceed" to send the request.
6. If there is a pending confirmation, ask for it naturally.
7. Do not sound robotic.

Context:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}
""".strip()

    return _stream_llm_text(prompt, temperature=0.25)


def lead_update_feedback_stream_to_writer(
    *,
    user_query: str,
    hydrated_lead: dict,
    field_updates: dict,
    invalid_fields: list[str],
    field_errors: dict,
    missing_fields: list[str],
    confirmation_resolution: str | None,
) -> str:
    safe_context = {
        "user_query": user_query,
        "current_details": {
            "name": hydrated_lead.get("name"),
            "email": hydrated_lead.get("email"),
            "phone": hydrated_lead.get("phone"),
            "preferred_contact_time": hydrated_lead.get("preferred_contact_time"),
            "apartment_id": hydrated_lead.get("apartment_id"),
        },
        "field_updates": field_updates,
        "invalid_fields": invalid_fields,
        "field_errors": field_errors,
        "missing_fields": missing_fields,
        "confirmation_resolution": confirmation_resolution,
    }

    prompt = f"""
You are Dorra's real-estate assistant.

Write one short natural user-facing reply.

Rules:
1. Be concise and natural.
2. Do not greet the user unless they greeted in this message.
3. If something failed validation, explain only that issue clearly.
4. If some fields were updated, mention that naturally.
5. If all required details are complete after the update, tell the user to say "proceed".
6. If details are still missing, mention only the missing items.
7. Do not mention internal field names.
8. Do not sound robotic.

Context:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}
""".strip()

    return _stream_llm_text(prompt, temperature=0.25)


def apartment_selection_stream_to_writer(apartment: dict, lead_data: dict) -> str:
    safe_context = {
        "selected_apartment": {
            "apartment_id": apartment.get("apartment_id"),
            "title": apartment.get("title"),
            "city": apartment.get("city"),
            "area": apartment.get("area"),
            "price": apartment.get("price"),
            "bedrooms": apartment.get("bedrooms"),
            "bathrooms": apartment.get("bathrooms"),
            "area_sqm": apartment.get("area_sqm"),
            "view": apartment.get("view"),
        },
        "known_lead_data": {
            "name": lead_data.get("name"),
            "email": lead_data.get("email"),
            "phone": lead_data.get("phone"),
            "preferred_contact_time": lead_data.get("preferred_contact_time"),
            "apartment_id": lead_data.get("apartment_id"),
        },
        "missing_fields": get_missing_fields(lead_data),
    }

    prompt = f"""
You are Dorra's real-estate assistant.

Write one short natural reply after the user selected an apartment.

Rules:
1. Confirm the apartment naturally.
2. Mention the apartment briefly using real details.
3. If details are missing, ask the user to send the missing details and say they can later say "proceed".
4. If details are already complete, tell the user they can say "proceed".
5. Be concise and smooth.

Context:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}
""".strip()

    return _stream_llm_text(prompt, temperature=0.25)


def send_success_stream_to_writer(lead_data: dict, apartment: dict) -> str:
    safe_context = {
        "lead_data": {
            "name": lead_data.get("name"),
            "email": lead_data.get("email"),
            "phone": lead_data.get("phone"),
            "preferred_contact_time": lead_data.get("preferred_contact_time"),
            "apartment_id": lead_data.get("apartment_id"),
        },
        "selected_apartment": {
            "apartment_id": apartment.get("apartment_id"),
            "title": apartment.get("title"),
            "city": apartment.get("city"),
            "area": apartment.get("area"),
        },
    }

    prompt = f"""
You are Dorra's real-estate assistant.

Write one short natural confirmation reply.

Rules:
1. Confirm that the request has been sent successfully.
2. Mention the selected apartment id.
3. Mention the preferred contact time if available.
4. Do not sound robotic.
5. Keep it concise.

Context:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}
""".strip()

    return _stream_llm_text(prompt, temperature=0.2)


def action_node(state):
    user_message = state["user_query"]
    pending_confirmation = dict(state.get("pending_confirmation", {}) or {})

    # pending confirmation guard
    if pending_confirmation:
        if is_yes_message(user_message):
            return {
                "action_result": {
                    "action": "update_lead_data",
                    "search_mode": "none",
                    "reference_type": "none",
                    "reference_value": None,
                    "field_updates": {
                        pending_confirmation.get("field"): pending_confirmation.get("suggested_value")
                    },
                    "invalid_fields": [],
                    "field_errors": {},
                    "confirmation_resolution": "accepted",
                    "source": "pending_confirmation_yes",
                },
                "action": "update_lead_data",
            }

        if is_no_message(user_message):
            return {
                "action_result": {
                    "action": "update_lead_data",
                    "search_mode": "none",
                    "reference_type": "none",
                    "reference_value": None,
                    "field_updates": {},
                    "invalid_fields": [],
                    "field_errors": {},
                    "confirmation_resolution": "rejected",
                    "source": "pending_confirmation_no",
                },
                "action": "update_lead_data",
            }

    planner_result = dict(plan_action_llm(user_message, state) or {})
    planner_action = normalize_planner_action(planner_result.get("action", "fallback_chat"))
    planner_result["action"] = planner_action

    # direct apartment id:
    # default => get details
    # override only if recent history strongly suggests compare/select/submit
    direct_apartment_id = extract_direct_apartment_id(user_message)
    if direct_apartment_id:
        id_intent_result = classify_id_intent_llm(
            user_message=user_message,
            apartment_id=direct_apartment_id,
            state=state,
        )
        id_action = str(id_intent_result.get("action") or "get_apartment_details").strip()

        allowed_id_actions = {
            "get_apartment_details",
            "select_apartment",
            "submit_lead",
            "analyze_shown_apartments",
        }
        if id_action not in allowed_id_actions:
            id_action = "get_apartment_details"

        return {
            "action_result": {
                "action": id_action,
                "search_mode": "none",
                "reference_type": "id",
                "reference_value": direct_apartment_id,
                "field_updates": {},
                "invalid_fields": [],
                "field_errors": {},
                "reply": "",
                "confirmation_resolution": None,
                "source": "direct_id_intent_llm",
                "id_intent_confidence": id_intent_result.get("confidence"),
                "id_intent_reason": id_intent_result.get("reason"),
            },
            "action": id_action,
        }

    # Let contact extraction run even when planner says reply_direct,
    # because short messages like "tomorrow at 3 pm" or "ahmed@gmail.com"
    # are easy for the planner to misclassify.
    if planner_action in {"update_lead_data", "submit_lead", "unsupported", "reply_direct"}:
        contact_result = extract_contact_updates_llm(user_message, state)

        valid_updates = dict(contact_result.get("valid_updates", {}) or {})
        invalid_fields = list(contact_result.get("invalid_fields", []) or [])
        field_errors = dict(contact_result.get("field_errors", {}) or {})

        if valid_updates or invalid_fields or field_errors:
            return {
                "action_result": {
                    "action": "update_lead_data",
                    "search_mode": "none",
                    "reference_type": planner_result.get("reference_type", "none"),
                    "reference_value": planner_result.get("reference_value"),
                    "field_updates": valid_updates,
                    "invalid_fields": invalid_fields,
                    "field_errors": field_errors,
                    "reply": planner_result.get("reply", ""),
                    "confirmation_resolution": planner_result.get("confirmation_resolution"),
                    "source": "planner_contact_update",
                },
                "action": "update_lead_data",
            }

    if planner_action == "unsupported":
        planner_action = "fallback_chat"
        planner_result["action"] = "fallback_chat"

    return {
        "action_result": planner_result,
        "action": planner_action,
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
    if len(matches) == 1:
        reply = stream_interest_hint(reply)

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
    reply = stream_interest_hint(reply)

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
    reply = stream_interest_hint(reply)

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

    lead_data = dict(state.get("lead_data", {}) or {})
    lead_data["apartment_id"] = apartment_id

    hydrated_lead = {
        "name": lead_data.get("name") or (state.get("user_profile") or {}).get("name"),
        "email": lead_data.get("email") or (state.get("user_profile") or {}).get("email"),
        "phone": lead_data.get("phone") or (state.get("user_profile") or {}).get("phone"),
        "preferred_contact_time": lead_data.get("preferred_contact_time") or (state.get("user_profile") or {}).get("preferred_contact_time"),
        "apartment_id": apartment_id,
    }

    reply = apartment_selection_stream_to_writer(apartment=apartment, lead_data=hydrated_lead)

    return {
        **build_focus_update(apartment),
        "lead_data": lead_data,
        "reply": reply,
        "stream_text": reply,
    }


def update_lead_data_node(state):
    action_result = state.get("action_result", {}) or {}
    field_updates = dict(action_result.get("field_updates", {}) or {})
    invalid_fields = list(action_result.get("invalid_fields", []) or [])
    field_errors = dict(action_result.get("field_errors", {}) or {})
    confirmation_resolution = action_result.get("confirmation_resolution")

    current_pending = dict(state.get("pending_confirmation", {}) or {})
    raw_user_message = str(state.get("user_query", "")).strip().lower()

    if current_pending and not confirmation_resolution:
        if is_yes_message(raw_user_message):
            field_updates = {
                current_pending.get("field"): current_pending.get("suggested_value")
            }
            confirmation_resolution = "accepted"
        elif is_no_message(raw_user_message):
            confirmation_resolution = "rejected"

    if not field_updates and not invalid_fields and not confirmation_resolution and not current_pending:
        contact_result = extract_contact_updates_llm(state.get("user_query", ""), state)
        field_updates = dict(contact_result.get("valid_updates", {}) or {})
        invalid_fields = list(contact_result.get("invalid_fields", []) or [])
        field_errors = dict(contact_result.get("field_errors", {}) or {})

    current_lead = dict(state.get("lead_data", {}) or {})
    user_profile = dict(state.get("user_profile", {}) or {})

    hydrated_before = {
        "name": current_lead.get("name") or user_profile.get("name"),
        "email": current_lead.get("email") or user_profile.get("email"),
        "phone": current_lead.get("phone") or user_profile.get("phone"),
        "preferred_contact_time": current_lead.get("preferred_contact_time") or user_profile.get("preferred_contact_time"),
        "preferred_contact_time_iso": current_lead.get("preferred_contact_time_iso") or user_profile.get("preferred_contact_time_iso"),
        "apartment_id": current_lead.get("apartment_id") or state.get("selected_apartment_id") or user_profile.get("apartment_id"),
    }

    missing_before = get_missing_fields(hydrated_before)

    merged_lead = {**hydrated_before, **field_updates}
    updated_profile = {**user_profile, **field_updates}

    # final addition: validate booking availability when user enters/modifies time
    if (
        merged_lead.get("preferred_contact_time_iso")
        and merged_lead.get("email")
        and merged_lead.get("apartment_id")
        and (
            "preferred_contact_time" in field_updates
            or "preferred_contact_time_iso" in field_updates
        )
    ):
        availability_result = validate_booking_availability_for_lead_time(state, merged_lead)

        if not availability_result.get("success"):
            conflict_type = str(availability_result.get("conflict_type") or "").strip().lower()

            # revert only the attempted time update
            merged_lead["preferred_contact_time"] = hydrated_before.get("preferred_contact_time")
            merged_lead["preferred_contact_time_iso"] = hydrated_before.get("preferred_contact_time_iso")
            updated_profile["preferred_contact_time"] = user_profile.get("preferred_contact_time")
            updated_profile["preferred_contact_time_iso"] = user_profile.get("preferred_contact_time_iso")

            field_updates.pop("preferred_contact_time", None)
            field_updates.pop("preferred_contact_time_iso", None)

            invalid_fields = list(dict.fromkeys(list(invalid_fields) + ["preferred_contact_time"]))

            if conflict_type == "user":
                message = (
                    "This time is not available because you already have another booking then. "
                    "You can modify your booking to another time, but you cannot view busy times or delete bookings here."
                )
            else:
                message = (
                    "This time is not available. "
                    "You can modify your booking to another time, but you cannot view busy times or delete bookings here."
                )

            field_errors["preferred_contact_time"] = {
                "reason": f"{conflict_type or 'booking'}_busy",
                "suggestion": None,
                "message": message,
            }

    missing_fields = get_missing_fields(merged_lead)
    just_completed = bool(missing_before) and not missing_fields

    pending_confirmation = {}
    if "email" in invalid_fields:
        email_error = field_errors.get("email", {}) or {}
        suggestion = email_error.get("suggestion")
        if suggestion:
            pending_confirmation = {
                "field": "email",
                "original_value": state.get("user_query", "").strip(),
                "suggested_value": suggestion,
                "type": "email_suggestion",
            }

    if just_completed:
        reply = lead_status_stream_to_writer(
            user_query=state.get("user_query", ""),
            lead_data=merged_lead,
            missing_fields=[],
            just_completed=True,
            pending_confirmation=bool(pending_confirmation),
        )
    elif invalid_fields or field_errors or field_updates or confirmation_resolution:
        reply = lead_update_feedback_stream_to_writer(
            user_query=state.get("user_query", ""),
            hydrated_lead=hydrated_before,
            field_updates=field_updates,
            invalid_fields=invalid_fields,
            field_errors=field_errors,
            missing_fields=missing_fields,
            confirmation_resolution=confirmation_resolution,
        )
    else:
        reply = lead_status_stream_to_writer(
            user_query=state.get("user_query", ""),
            lead_data=merged_lead,
            missing_fields=missing_fields,
            just_completed=False,
            pending_confirmation=bool(pending_confirmation),
        )

    return {
        "lead_data": merged_lead,
        "user_profile": updated_profile,
        "missing_fields": missing_fields,
        "pending_confirmation": pending_confirmation if confirmation_resolution != "accepted" else {},
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
        "preferred_contact_time_iso": lead_data.get("preferred_contact_time_iso") or user_profile.get("preferred_contact_time_iso"),
        "apartment_id": lead_data.get("apartment_id") or state.get("selected_apartment_id"),
    }

    missing_fields = get_missing_fields(hydrated_lead)

    return {
        "lead_data": hydrated_lead,
        "missing_fields": missing_fields,
    }


def missing_lead_info_node(state):
    lead_data = state.get("lead_data", {}) or {}
    missing_fields = state.get("missing_fields", []) or []

    reply = lead_status_stream_to_writer(
        user_query=state.get("user_query", ""),
        lead_data=lead_data,
        missing_fields=missing_fields,
        just_completed=False,
        pending_confirmation=bool(state.get("pending_confirmation")),
    )

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

    agent_email = str(selected_apartment.get("agent_email") or "").strip().lower()
    requested_contact_at_iso = str(lead_data.get("preferred_contact_time_iso") or "").strip()
    lead_email = str(lead_data.get("email") or "").strip().lower()

    if not requested_contact_at_iso:
        reply = "I need a valid contact time before I can submit your request."
        return {
            "reply": reply,
            "stream_text": reply,
        }

    if not lead_email:
        reply = "I need a valid email before I can submit your request."
        return {
            "reply": reply,
            "stream_text": reply,
        }

    reservation = reserve_booking_time_slot(
        agent_email=agent_email,
        user_email=lead_email,
        requested_contact_at_iso=requested_contact_at_iso,
        apartment_id=apartment_id,
    )

    if not reservation.get("success"):
        conflict_type = str(reservation.get("conflict_type") or "").strip().lower()

        if conflict_type == "user":
            reply = (
                "This time is not available because you already have another booking then. "
                "You can modify your booking to another time, but you cannot view busy times or delete bookings here."
            )
        else:
            reply = (
                "This time is not available. "
                "You can modify your booking to another time, but you cannot view busy times or delete bookings here."
            )

        return {
            "reply": reply,
            "stream_text": reply,
        }

    result = send_email(selected_apartment, lead_data)

    if result.get("success"):
        remembered_profile = dict(state.get("user_profile", {}) or {})
        remembered_profile.update(
            {
                "name": lead_data.get("name") or remembered_profile.get("name"),
                "email": lead_data.get("email") or remembered_profile.get("email"),
                "phone": lead_data.get("phone") or remembered_profile.get("phone"),
                "preferred_contact_time": lead_data.get("preferred_contact_time") or remembered_profile.get("preferred_contact_time"),
                "preferred_contact_time_iso": lead_data.get("preferred_contact_time_iso") or remembered_profile.get("preferred_contact_time_iso"),
            }
        )

        reply = send_success_stream_to_writer(lead_data, selected_apartment)

        return {
            "reply": reply,
            "stream_text": reply,
            "lead_data": {},
            "missing_fields": [],
            "user_profile": remembered_profile,
            "pending_confirmation": {},
        }

    reply = result.get("message", "Failed to send email.")
    return {
        "reply": reply,
        "stream_text": reply,
    }



def reply_direct_node(state):
    action_result = state.get("action_result", {}) or {}
    reply = str(action_result.get("reply") or "").strip()

    if not reply:
        reply = fallback_chat_stream_to_writer(state.get("user_query", ""), state)
        return {
            "reply": reply,
            "stream_text": reply,
        }

    writer = get_stream_writer()
    writer(reply)
    return {
        "reply": reply,
        "stream_text": reply,
    }


def general_chat_node(state):
    user_message = state["user_query"]
    reply = general_chat_stream_to_writer(user_message, state)
    return {"reply": reply, "stream_text": reply}


def fallback_chat_node(state):
    user_message = state["user_query"]
    reply = fallback_chat_stream_to_writer(user_message, state)
    return {"reply": reply, "stream_text": reply}


def unsupported_node(state):
    reply = (
        "I can help with apartment search, apartment details, selecting an apartment, "
        "updating your contact details, lead requests, or Dorra company information only.\n"
        "Try asking for a property, asking about one of the shown options, "
        "sharing your contact details, or asking about Dorra.\n"
        "Or you can contact one of our sales team via Hotline: 16077 or Email: info@dorra.com"
    )
    return {"reply": reply, "stream_text": reply}


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
    graph.add_node("reply_direct", reply_direct_node)
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
            "reply_direct": "reply_direct",
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
    graph.add_edge("reply_direct", END)
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