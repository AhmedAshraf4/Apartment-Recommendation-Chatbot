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


def classify_routing_hint_llm(user_message: str, state: dict) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    safe_context = {
        "recent_history": (state.get("chat_history") or [])[-6:],
        "last_search_filters": state.get("last_search_filters") or {},
        "selected_apartment": {
            "apartment_id": (state.get("selected_apartment") or {}).get("apartment_id"),
            "title": (state.get("selected_apartment") or {}).get("title"),
            "city": (state.get("selected_apartment") or {}).get("city"),
            "area": (state.get("selected_apartment") or {}).get("area"),
        },
        "shown_apartments": [
            {
                "order": i + 1,
                "apartment_id": apt.get("apartment_id"),
                "title": apt.get("title"),
                "city": apt.get("city"),
                "area": apt.get("area"),
            }
            for i, apt in enumerate((state.get("last_shown_apartments") or [])[:5])
        ],
    }

    prompt = f"""
    You are a strict routing hint classifier for a real-estate assistant.

    Return JSON only.
    Do not explain anything.

    Classify the user's message into one of:
    - "search"
    - "selected_followup"
    - "neither"

    Definitions:
    - "search": the user is asking for a new property search, refining the previous search, or shifting the search to a different location, property type, budget, size, bedrooms, bathrooms, or listing set.
    - "selected_followup": the user is asking about the currently selected apartment using pronouns like it/this/that, with no new explicit apartment reference and no new search criteria.
    - "neither": neither of the above.

    Interpret by meaning, not exact grammar.
    The user may write short, incomplete, or broken English.

    Rules:
    1. If the user clearly asks for a new property type, location, bedrooms, bathrooms, price, size, or sorting/filtering, return "search".
    2. Treat short search-shift messages as "search" too, for example:
       - "october?"
       - "what about october?"
       - "options in october?"
       - "options is october?"
       - "zayed instead"
       - "new cairo?"
       - "villa options?"
       - "cheaper ones?"
       - "3 bedrooms?"
       - "under 4 million?"
       - "sort by price"
    3. Amenity questions about the currently shown apartments are NOT "search". They are "neither". Examples:
       - "pool?"
       - "gym?"
       - "parking?"
       - "which of the above have a pool?"
       - "which one has gym?"
       - "any with clubhouse?"
    4. If the user is asking about the currently selected apartment with pronouns and no new search filter, return "selected_followup". Examples:
       - "does it have a pool?"
       - "how much is it?"
       - "is it larger?"
       - "what about its view?"
    5. If the message is about the shown list as a group, not the selected apartment, return "neither".
    6. If uncertain between "search" and "neither":
       - choose "search" for location/property/filter changes
       - choose "neither" for amenity/comparison questions about current results
    7. If uncertain between "selected_followup" and "neither", choose "selected_followup" only when the message is clearly about the currently focused apartment.

    Return exactly:
    {{
      "hint": "neither"
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
        return {"route_hint": "neither"}

    route_hint = str(parsed.get("route_hint", "neither")).strip().lower()
    if route_hint not in {"search", "selected_followup", "neither"}:
        route_hint = "neither"

    return {"route_hint": route_hint}