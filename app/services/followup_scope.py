import json
import re
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


def classify_followup_scope(user_message: str, state: dict) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    safe_context = {
        "recent_history": (state.get("chat_history") or [])[-6:],
        "shown_apartments": [
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
            }
            for i, apt in enumerate((state.get("last_shown_apartments") or [])[:5])
        ],
        "selected_apartment": {
            "apartment_id": (state.get("selected_apartment") or {}).get("apartment_id"),
            "title": (state.get("selected_apartment") or {}).get("title"),
            "city": (state.get("selected_apartment") or {}).get("city"),
            "area": (state.get("selected_apartment") or {}).get("area"),
            "price": (state.get("selected_apartment") or {}).get("price"),
        },
    }

    prompt = f"""
    You are a strict routing helper for a real-estate assistant.

    Return JSON only.
    Do not explain anything.
    Do not output markdown.

    Choose one value for "scope":
    - "shown_list" -> the user is asking about the currently shown apartments as a group
    - "selected_apartment" -> the user is asking about the currently focused apartment
    - "new_search" -> the user is asking for a new apartment search or changing the search criteria/listing set
    - "none" -> unclear / none of the above

    Interpret the user's message by meaning, not by exact grammar.
    The user may write short, broken, or informal English.

    Rules:
    1. Amenity questions about the currently shown apartments should be "shown_list".
    2. Examples that should usually be "shown_list":
       - "pool?"
       - "gym?"
       - "parking?"
       - "which of the above have a pool?"
       - "which one has gym?"
       - "any with clubhouse?"
       - "which options have garden view?"
    3. Pronoun follow-ups like "does it have a pool?", "how much is it?", "is it larger?" should usually be "selected_apartment" if a selected apartment exists.
    4. Explicit new requests like "i want a townhouse in zayed" should be "new_search".
    5. Messages should be "new_search" only when the user is changing or requesting a new search axis such as:
       - city or location
       - property type
       - price/budget
       - bedrooms/bathrooms
       - size
       - sorting
    6. Examples that should usually be "new_search":
       - "october?"
       - "options in october?"
       - "options is october?"
       - "what about october?"
       - "zayed instead"
       - "new cairo?"
       - "villa options?"
       - "cheaper ones?"
       - "3 bedrooms?"
       - "under 4 million?"
       - "sort by price"
    7. If the user refers to "above", "these", "those", "the options", or asks about the current options as a group, choose "shown_list" unless they are clearly changing the search criteria.
    8. If the user is clearly continuing discussion about one focused apartment, choose "selected_apartment".
    9. Do not classify amenity-only questions as "new_search" unless the user clearly asks for a fresh search based on that amenity.
    10. If uncertain between "shown_list" and "new_search":
       - choose "shown_list" for amenity/comparison questions about current results
       - choose "new_search" for location/filter/property-type changes

    Return exactly:
    {{
      "scope": "none"
    }}

    Safe context:
    {json.dumps(safe_context, ensure_ascii=False, indent=2)}

    User message:
    {user_message}
    """.strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = parse_json(raw)

    if not isinstance(parsed, dict):
        return {"scope": "none"}

    scope = str(parsed.get("scope", "none")).strip().lower()
    if scope not in {"shown_list", "selected_apartment", "new_search", "none"}:
        scope = "none"

    return {"scope": scope}