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


def resolve_search_history_reference_llm(user_message: str, state: dict) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    search_history = list(state.get("search_history", []) or [])
    active_index = state.get("active_search_history_index")

    summarized_history = []
    for i, item in enumerate(search_history):
        filters = item.get("filters") or {}
        matches = item.get("matches") or []

        cities = sorted(
            {
                str(apt.get("city") or "").strip().lower()
                for apt in matches
                if str(apt.get("city") or "").strip()
            }
        )

        areas = sorted(
            {
                str(apt.get("area") or "").strip().lower()
                for apt in matches
                if str(apt.get("area") or "").strip()
            }
        )

        summarized_history.append(
            {
                "history_index": i,
                "is_active": i == active_index,
                "user_query": item.get("user_query"),
                "created_at_iso": item.get("created_at_iso"),
                "filters": filters,
                "results_count": item.get("results_count"),
                "cities": cities[:5],
                "areas": areas[:8],
            }
        )

    prompt = f"""
    You resolve whether the user is asking to use an older saved search.

    Return JSON only.
    Do not explain anything.
    Do not output markdown.

    Rules:
    1. The CURRENT active search is the default.
    2. Ordinals like "first", "second", "this one", "that one" refer to the CURRENT shown apartments unless the user indicates another saved search.
    3. The user can indicate another saved search in either of these ways:
       - explicit old-search wording:
         - "old search"
         - "previous search"
         - "earlier results"
         - "go back to the zayed results"
       - mentioning a location/area/city that matches a saved search different from the current active search:
         - "the first one in zayed"
         - "give me the october ones"
         - "what about the new cairo results"
         - "the first one from zayed"
    4. If the user mentions a location that matches the CURRENT active search, do not restore.
    5. If the user mentions a location that matches exactly one older saved search, restore that one.
    6. If the user mentions a location and also uses apartment references like "first one", "second one", "this one", treat that as a strong signal to restore the matching saved search before apartment resolution.
    7. Only return should_restore=false when the user is clearly staying on the current search or no older search matches.

    Return exactly this schema:
    {{
      "should_restore": false,
      "history_index": null,
      "reason": "not referring to another saved search"
    }}

    Search history:
    {json.dumps(summarized_history, ensure_ascii=False, indent=2)}

    User message:
    {user_message}
    """.strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = _safe_json(raw) or {}

    should_restore = bool(parsed.get("should_restore", False))
    history_index = parsed.get("history_index")

    if not isinstance(history_index, int):
        history_index = None

    if history_index is None or history_index < 0 or history_index >= len(search_history):
        should_restore = False
        history_index = None

    return {
        "should_restore": should_restore,
        "history_index": history_index,
        "reason": parsed.get("reason"),
    }