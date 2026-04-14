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
- "search": the user is explicitly asking for a new property search or refining the previous search.
- "selected_followup": the user is asking about the currently selected apartment using pronouns like it/this/that, with no new explicit apartment reference.
- "neither": neither of the above.

Rules:
1. If the user clearly asks for a new property type, location, bedrooms, bathrooms, price, or a fresh listing request, return "search".
2. If the user is clearly asking about the currently selected apartment, return "selected_followup".
3. If the user explicitly mentions "first", "second", apartment ID, or another new option, do NOT return "selected_followup".
4. Prefer "search" when the user clearly changes the requested property type, such as apartment -> townhouse.
5. Return exactly this shape:

{{
  "route_hint": "neither"
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