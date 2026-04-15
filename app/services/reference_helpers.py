import json

from langchain_openai import ChatOpenAI

from app.core.config import settings


def build_apartment_reference_map(apartments):
    """
    Kept only for compatibility with the rest of the codebase.
    Reference resolution is now handled by the LLM using session context.
    """
    return {}


def get_apartment_by_id(apartments, apartment_id):
    if not apartment_id:
        return None

    apartment_id = str(apartment_id).strip().lower()

    for apartment in apartments or []:
        current_id = str(apartment.get("apartment_id", "")).strip().lower()
        if current_id == apartment_id:
            return apartment

    return None


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


def resolve_apartment_reference_llm(user_query: str, state: dict) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    shown_apartments = [
        {
            "order": i + 1,
            "apartment_id": apt.get("apartment_id"),
            "title": apt.get("title"),
            "city": apt.get("city"),
            "area": apt.get("area"),
            "bedrooms": apt.get("bedrooms"),
            "bathrooms": apt.get("bathrooms"),
            "area_sqm": apt.get("area_sqm"),
            "view": apt.get("view"),
            "price": apt.get("price"),
            "amenities": apt.get("amenities"),
            "description": apt.get("description"),
        }
        for i, apt in enumerate((state.get("last_shown_apartments") or [])[:10])
    ]

    selected = state.get("selected_apartment") or {}

    context = {
        "latest_user_message": user_query,
        "recent_history": (state.get("chat_history") or [])[-10:],
        "shown_apartments": shown_apartments,
        "selected_apartment": {
            "apartment_id": selected.get("apartment_id"),
            "title": selected.get("title"),
            "city": selected.get("city"),
            "area": selected.get("area"),
            "bedrooms": selected.get("bedrooms"),
            "bathrooms": selected.get("bathrooms"),
            "area_sqm": selected.get("area_sqm"),
            "view": selected.get("view"),
            "price": selected.get("price"),
        },
    }

    prompt = f"""
You resolve apartment references in a real-estate chatbot.

Return JSON only.
Do not explain anything.
Do not output markdown.

Your task:
Decide whether the user's latest message refers to:
1. one apartment from the CURRENTLY SHOWN apartment list
2. the CURRENTLY SELECTED apartment
3. no apartment / unclear

Use:
- the latest user message
- recent chat history
- shown apartments
- selected apartment

Important behavior rules:
1. Interpret references by meaning, not rigid keyword matching.
2. References may include messages such as:
   - "1"
   - "2"
   - "first"
   - "second option"
   - "the first one"
   - "the last one"
   - "the option in the list"
   - "the other one"
   - "this one"
   - "that one"
   - "the one from the 5 apartments"
3. If the user explicitly contrasts the shown list with the selected apartment, prefer the shown list.
4. If the user says things like "in the list", "from the list", "among the 5 apartments", or similar, prefer shown_list.
5. If the user is clearly continuing a focused discussion on one apartment, selected_apartment may be correct.
6. If unclear, return scope = "none".

Return exactly this schema:
{{
  "scope": "shown_list",
  "apartment_id": "ap022",
  "confidence": 0.95,
  "reason": "user explicitly referred to the last option in the shown list"
}}

Allowed scope values:
- "shown_list"
- "selected_apartment"
- "none"

Rules:
- If scope is "none", apartment_id must be null.
- If scope is "selected_apartment", apartment_id should be the selected apartment id.
- If scope is "shown_list", apartment_id must be one of the shown apartment ids.
- Confidence must be a number between 0 and 1.

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = _safe_json(raw)

    if not isinstance(parsed, dict):
        return {
            "scope": "none",
            "apartment_id": None,
            "confidence": 0.0,
            "reason": "invalid_llm_output",
        }

    scope = str(parsed.get("scope") or "none").strip().lower()
    apartment_id = parsed.get("apartment_id")
    confidence = parsed.get("confidence", 0.0)
    reason = str(parsed.get("reason") or "").strip()

    if scope not in {"shown_list", "selected_apartment", "none"}:
        scope = "none"

    shown_ids = {
        str(apt.get("apartment_id")).strip().lower()
        for apt in shown_apartments
        if apt.get("apartment_id")
    }

    selected_id = str(selected.get("apartment_id") or "").strip().lower()
    apartment_id_norm = str(apartment_id).strip().lower() if apartment_id else None

    if scope == "shown_list":
        if not apartment_id_norm or apartment_id_norm not in shown_ids:
            return {
                "scope": "none",
                "apartment_id": None,
                "confidence": 0.0,
                "reason": "llm_returned_invalid_shown_list_id",
            }

    elif scope == "selected_apartment":
        if not selected_id:
            return {
                "scope": "none",
                "apartment_id": None,
                "confidence": 0.0,
                "reason": "no_selected_apartment_available",
            }
        apartment_id_norm = selected_id

    else:
        apartment_id_norm = None

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "scope": scope,
        "apartment_id": apartment_id_norm,
        "confidence": confidence,
        "reason": reason,
    }


def resolve_apartment_reference(user_query, state):
    result = resolve_apartment_reference_llm(user_query, state)
    return result.get("apartment_id")


def is_selection_request(user_query):
    return False


def is_detail_request(user_query):
    return False


def render_apartment_details(apartment):
    if not apartment:
        return (
            "I could not find that apartment in the current session. "
            "Please search again or mention the apartment ID."
        )

    bedrooms = apartment.get("bedrooms", "N/A")
    bathrooms = apartment.get("bathrooms", "N/A")

    return (
        f"Here are the details for apartment {apartment.get('apartment_id', 'N/A')}:\n\n"
        f"Type: {apartment.get('title', 'N/A')}\n"
        f"Price: {apartment.get('price', 'N/A')} EGP\n"
        f"Location: {apartment.get('city', 'N/A')} - {apartment.get('area', 'N/A')}\n"
        f"Specs: {bedrooms} bedrooms, {bathrooms} bathrooms, {apartment.get('area_sqm', 'N/A')} sqm\n"
        f"View: {apartment.get('view', 'N/A')}\n"
        f"Amenities: {apartment.get('amenities', 'N/A')}\n"
        f"Description: {apartment.get('description', 'N/A')}\n\n"
        f"If you want to continue with this unit, send me its ID or tell me you'd like to proceed with this apartment."
    )


def answer_apartment_followup(user_query, apartment, question_focus="none"):
    if not apartment:
        return (
            "I could not find that apartment in the current session. "
            "Please search again or mention the apartment ID."
        )

    apartment_id = apartment.get("apartment_id", "N/A")
    amenities = str(apartment.get("amenities", "") or "")
    description = str(apartment.get("description", "") or "")
    view = str(apartment.get("view", "") or "")
    city = str(apartment.get("city", "") or "")
    area_name = str(apartment.get("area", "") or "")
    searchable_text = " ".join([amenities, description, view]).lower()

    if question_focus == "amenity_pool":
        return (
            f"Yes, apartment {apartment_id} appears to mention a pool."
            if "pool" in searchable_text
            else f"I do not see a pool mentioned for apartment {apartment_id} in the available details."
        )

    if question_focus == "amenity_gym":
        return (
            f"Yes, apartment {apartment_id} appears to mention a gym."
            if "gym" in searchable_text
            else f"I do not see a gym mentioned for apartment {apartment_id} in the available details."
        )

    if question_focus == "amenity_parking":
        return (
            f"Yes, apartment {apartment_id} appears to mention parking."
            if "parking" in searchable_text
            else f"I do not see parking mentioned for apartment {apartment_id} in the available details."
        )

    if question_focus == "amenity_garden":
        return (
            f"Yes, apartment {apartment_id} appears to mention a garden."
            if "garden" in searchable_text
            else f"I do not see a garden mentioned for apartment {apartment_id} in the available details."
        )

    if question_focus == "price":
        return f"Apartment {apartment_id} is priced at {apartment.get('price', 'N/A')} EGP."

    if question_focus == "bedrooms":
        return f"Apartment {apartment_id} has {apartment.get('bedrooms', 'N/A')} bedrooms."

    if question_focus == "bathrooms":
        return f"Apartment {apartment_id} has {apartment.get('bathrooms', 'N/A')} bathrooms."

    if question_focus == "area":
        return f"Apartment {apartment_id} has an area of {apartment.get('area_sqm', 'N/A')} sqm."

    if question_focus == "view":
        return f"Apartment {apartment_id} has view: {apartment.get('view', 'N/A')}."

    if question_focus == "amenities":
        return f"Apartment {apartment_id} has these amenities listed: {apartment.get('amenities', 'N/A')}."

    if question_focus == "description":
        return f"Apartment {apartment_id} description: {apartment.get('description', 'N/A')}."

    if question_focus == "location":
        return f"Apartment {apartment_id} is located in {city} - {area_name}."

    return render_apartment_details(apartment)