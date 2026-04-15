import json
import re
from typing import Any
from langchain_openai import ChatOpenAI

from app.core.config import settings


def parse_json(text):
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def build_action_context(state: dict[str, Any]) -> dict[str, Any]:
    recent_history = (state.get("chat_history") or [])[-8:]
    last_shown = (state.get("last_shown_apartments") or [])[:5]
    selected = state.get("selected_apartment") or {}
    lead_data = state.get("lead_data") or {}
    user_profile = state.get("user_profile") or {}

    shown_apartments = []
    for index, apartment in enumerate(last_shown, start=1):
        shown_apartments.append(
            {
                "order": index,
                "apartment_id": apartment.get("apartment_id"),
                "title": apartment.get("title"),
                "city": apartment.get("city"),
                "area": apartment.get("area"),
                "price": apartment.get("price"),
                "bedrooms": apartment.get("bedrooms"),
                "bathrooms": apartment.get("bathrooms"),
                "area_sqm": apartment.get("area_sqm"),
                "view": apartment.get("view"),
                "amenities": apartment.get("amenities"),
                "description": apartment.get("description"),
            }
        )

    return {
        "recent_history": recent_history,
        "last_search_filters": state.get("last_search_filters") or {},
        "shown_apartments": shown_apartments,
        "selected_apartment": {
            "apartment_id": selected.get("apartment_id"),
            "title": selected.get("title"),
            "city": selected.get("city"),
            "area": selected.get("area"),
            "price": selected.get("price"),
        },
        "lead_data": {
            "name": lead_data.get("name"),
            "email": lead_data.get("email"),
            "phone": lead_data.get("phone"),
            "preferred_contact_time": lead_data.get("preferred_contact_time"),
            "apartment_id": lead_data.get("apartment_id"),
        },
        "user_profile": {
            "name": user_profile.get("name"),
            "email": user_profile.get("email"),
            "phone": user_profile.get("phone"),
            "preferred_contact_time": user_profile.get("preferred_contact_time"),
        },
    }


def normalize_action_output(data):
    if not isinstance(data, dict):
        return {
            "action": "unsupported",
            "search_mode": "none",
            "reference_type": "none",
            "reference_value": None,
            "field_updates": {},
        }

    action = str(data.get("action", "unsupported")).strip().lower()
    if action not in {
        "general_chat",
        "company_info",
        "search",
        "get_apartment_details",
        "select_apartment",
        "analyze_shown_apartments",
        "update_lead_data",
        "submit_lead",
        "fallback_chat",
        "unsupported",
    }:
        action = "unsupported"

    search_mode = str(data.get("search_mode", "none")).strip().lower()
    if search_mode not in {"none", "new", "refine"}:
        search_mode = "none"

    reference_type = str(data.get("reference_type", "none")).strip().lower()
    if reference_type not in {"none", "ordinal", "id", "selected"}:
        reference_type = "none"

    reference_value = data.get("reference_value")
    if reference_value is not None:
        reference_value = str(reference_value).strip()

    field_updates = data.get("field_updates", {})
    if not isinstance(field_updates, dict):
        field_updates = {}

    clean_updates = {}
    for key in ["name", "email", "phone", "preferred_contact_time"]:
        value = field_updates.get(key)
        if value is not None and str(value).strip():
            clean_updates[key] = str(value).strip()

    return {
        "action": action,
        "search_mode": search_mode,
        "reference_type": reference_type,
        "reference_value": reference_value,
        "field_updates": clean_updates,
    }


def detect_action(user_query: str, state: dict[str, Any]) -> dict[str, Any]:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    context = build_action_context(state)

    prompt = f"""
You are a strict action planner for Dorra's apartment assistant.

Return JSON only.
Do not return markdown.
Do not explain your reasoning.

Allowed actions:
- "general_chat"
- "company_info"
- "search"
- "get_apartment_details"
- "select_apartment"
- "analyze_shown_apartments"
- "update_lead_data"
- "submit_lead"
- "fallback_chat"
- "unsupported"

Allowed search_mode:
- "none"
- "new"
- "refine"

Allowed reference_type:
- "none"
- "ordinal"
- "id"
- "selected"

Rules:
1. Use "general_chat" for greetings, thanks, bye, and conversational filler.
2. Use "company_info" for questions specifically about Dorra as a company.
3. Use "search" for fresh apartment searches and search refinements.
4. Use "get_apartment_details" when the user asks about one specific shown apartment or selected apartment, including pronouns like "it", "this one", or "that one".
5. Use "select_apartment" when the user is choosing one specific shown apartment.
6. Use "analyze_shown_apartments" when the user is asking about the currently shown list as a group.
7. Use "update_lead_data" when the user provides or changes name, email, phone, or preferred contact time.
8. Use "submit_lead" when the user clearly wants to proceed, continue, submit, follow up, or request contact using the current apartment and lead details.
9. If the user explicitly refers to "first", "second", "third", etc., set reference_type="ordinal" and reference_value to that word or number.
10. If the user mentions an apartment ID, set reference_type="id".
11. If the user says "it", "this one", "that one", or similar and there is a selected apartment in context, set reference_type="selected".
12. Use "analyze_shown_apartments" for questions like:
   - "which one is the most expensive?"
   - "which of the above have a pool?"
   - "which one is better?"
   - "which is bigger?"
   - "which options have 3 bedrooms?"
   - "which of these are in zayed?"
   - "compare the shown apartments"
13. If the user explicitly asks for a new property search or clearly changes the requested property type, area, or search constraints, prefer "search" instead of continuing the previous apartment discussion.
14. Use "fallback_chat" when the request is conversationally tied to the current apartment discussion but does not fit cleanly into the other actions.
15. For update_lead_data, extract only:
   - name
   - email
   - phone
   - preferred_contact_time
16. Bedroom-count, bathroom-count, budget, size, location, and property-type changes should be treated as "search", even when short.
    Examples:
    - "3 bedrooms?"
    - "which options have 3 bedrooms?"
    - "2 bathrooms?"
    - "under 4 million?"
    - "october?"
    - "villas instead"

Return exactly this JSON shape:
{{
  "action": "unsupported",
  "search_mode": "none",
  "reference_type": "none",
  "reference_value": null,
  "field_updates": {{}}
}}

Conversation context:
{json.dumps(context, ensure_ascii=False, indent=2)}

User message:
{user_query}
""".strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = parse_json(raw)
    return normalize_action_output(parsed)