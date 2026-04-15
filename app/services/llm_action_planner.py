import json

from langchain_openai import ChatOpenAI

from app.core.config import settings


def build_planner_context(state: dict) -> dict:
    return {
        "user_query": state.get("user_query"),
        "chat_history": (state.get("chat_history") or [])[-12:],
        "selected_apartment_id": state.get("selected_apartment_id"),
        "selected_apartment": state.get("selected_apartment"),
        "last_shown_apartments": [
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
            for i, apt in enumerate(state.get("last_shown_apartments") or [])
        ],
        "lead_data": state.get("lead_data") or {},
        "user_profile": state.get("user_profile") or {},
        "pending_confirmation": state.get("pending_confirmation") or {},
        "pending_restricted_update": state.get("pending_restricted_update") or {},
        "last_search_filters": state.get("last_search_filters") or {},
    }


def parse_json(text: str):
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


def _normalize_reference_type(value: str) -> str:
    value = str(value or "").strip().lower()
    if value not in {"none", "id", "ordinal", "selected"}:
        return "none"
    return value


def _normalize_action(value: str) -> str:
    value = str(value or "").strip()
    allowed = {
        "search",
        "get_apartment_details",
        "select_apartment",
        "analyze_shown_apartments",
        "update_lead_data",
        "submit_lead",
        "company_info",
        "reply_direct",
        "fallback_chat",
    }
    if value not in allowed:
        return "fallback_chat"
    return value


def _normalize_field_updates(value) -> dict:
    if not isinstance(value, dict):
        return {}

    allowed_keys = {"name", "email", "phone", "preferred_contact_time"}
    cleaned = {}

    for key, raw_value in value.items():
        key = str(key or "").strip()
        if key not in allowed_keys:
            continue

        if raw_value is None:
            continue

        text = str(raw_value).strip()
        if not text:
            continue

        cleaned[key] = text

    return cleaned


def plan_action_llm(user_message: str, state: dict) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    context = build_planner_context(state)

    prompt = f"""
    You are the main decision planner for Dorra's real-estate assistant.

    Your job is to choose the BEST next action using:
    - the latest user message
    - recent chat history
    - the currently selected apartment
    - the currently shown apartments
    - saved user details
    - pending confirmations
    - previous search filters
    - the ongoing lead/contact flow

    You are NOT a keyword matcher.
    You are a context-aware planner.
    Use the session history heavily.

    Return JSON only.
    Do not explain anything.
    Do not output markdown fences.
    Do not output extra text.

    Allowed actions:
    - "search"
    - "get_apartment_details"
    - "select_apartment"
    - "analyze_shown_apartments"
    - "update_lead_data"
    - "submit_lead"
    - "company_info"
    - "reply_direct"
    - "fallback_chat"

    What each action means:

    1. "search"
    Use when the user wants a fresh property search or a refinement/change of the current search.
    This includes changing:
    - city or location
    - property type
    - budget
    - bedrooms or bathrooms
    - size
    - sorting
    - required amenity, when the user is asking for a fresh set of listings

    Examples:
    - show me apartments in new cairo
    - i want something cheaper
    - find me 3 bedrooms in zayed
    - show me townhouses instead
    - under 8 million
    - sort by area descending
    - what about october?
    - october?
    - options in october?
    - options is october?
    - zayed instead
    - villas instead
    - townhouses in zayed
    - something with pool in zayed

    2. "get_apartment_details"
    Use when the user is asking about one specific apartment.
    This includes:
    - direct apartment IDs
    - ordinal references like first / second / third
    - selected apartment references
    - pronouns like it / this one / that one when one apartment is clearly in focus
    - amenity/detail questions about one focused apartment

    Examples:
    - tell me more about the second one
    - does it have a pool
    - what is the price of this one
    - details about ap003
    - what about its view
    - does this one have gym

    3. "select_apartment"
    Use when the user is choosing one specific apartment.

    This includes:
    - choosing by pronoun when one apartment is clearly in focus
    - choosing by ordinal from a shown list
    - choosing by explicit apartment ID
    - switching from one apartment to another by explicitly naming a different apartment

    Examples:
    - i want this one
    - i want this
    - i want it
    - i choose the second one
    - go with ap003
    - use this apartment
    - i’ll take the first one
    - this is the one i want
    - i want the selected one
    - i want ap023
    - i choose ap023
    - go with ap023
    - use ap023
    - i want apartment ap023

    4. "analyze_shown_apartments"
    Use when the user is asking about the currently shown apartments as a group.
    This includes amenity/comparison/filtering questions about the CURRENT shown set, when the user is not asking for a fresh search.

    Examples:
    - which one is cheaper
    - compare them
    - which of these has the best view
    - among these which is bigger
    - which ones have 3 bedrooms
    - what are the differences
    - pool?
    - gym?
    - parking?
    - which of these have a pool?
    - which one has gym?
    - any with clubhouse?

    5. "update_lead_data"
    Use when the user is sharing, correcting, confirming, or insisting about contact details.
    This includes:
    - name
    - email
    - phone
    - preferred contact time
    This also includes continuing a blocked validation discussion.

    Examples:
    - my phone is 010...
    - my email is ...
    - call me tomorrow at 10
    - no that email is wrong
    - yes use that email
    - i insist on this time
    - keep 10 am
    - use my old number

    6. "submit_lead"
    Use when the user is clearly asking to move forward with the lead/contact request.

    Examples:
    - proceed
    - continue
    - send my request
    - contact them
    - book it
    - submit
    - go ahead

    7. "company_info"
    Use only for Dorra company questions.

    Examples:
    - what is dorra's hotline
    - where is your office
    - tell me about dorra
    - who is dorra
    - who is the ceo
    - crc dorra
    - dorra

    8. "reply_direct"
    Use only for short conversational glue INSIDE the real-estate flow.
    This is for short natural replies that help continue the apartment conversation,
    not for unrelated topic changes and not for open-domain chatting.

    Examples:
    - hi
    - thanks
    - what was that
    - why
    - okay
    - can you explain
    - no i meant the other thing
    - i do not like this method
    - what do you need from me now

    Do NOT use "reply_direct" for:
    - new property requests
    - apartment comparisons/group analysis
    - apartment detail questions
    - contact details
    - unrelated topic switches like cooking, brands, sports, coding, or general knowledge

    9. "fallback_chat"
    Use when the message is conversationally related to the current apartment flow,
    but none of the other actions fits well enough.

    High-priority planning rules:
    1. Prefer conversation understanding over literal phrase matching.
    2. Resolve short follow-ups from history, not in isolation.
    3. Resolve pronouns like "it", "this one", "that one", "the other one", "second one" from context.
    4. If the user is in a contact-update flow, do NOT force apartment search/details unless the latest message clearly changes the property request.
    5. If the user is in a blocked-time or invalid-email discussion, keep the conversation in "update_lead_data".
    6. If the user is choosing among currently shown options, prefer "select_apartment" or "analyze_shown_apartments" instead of "search".
    7. If the user is asking about one apartment after a shown list, prefer "get_apartment_details".
    8. Use "reply_direct" only for greetings, acknowledgments, confusion, small clarifications, objections, or short conversational glue inside the supported flow.
    9. Do NOT use "reply_direct" for unrelated topic switches. If the user is switching to a non-apartment, non-lead, non-Dorra topic, prefer "fallback_chat" rather than acting like a general chatbot.
    10. Do not force "submit_lead" unless the user is clearly ready to move forward.
    11. Do not force "company_info" unless the question is actually about Dorra as a company.
    12. When the user says things like "the second one", "first one", "option 3", set reference_type="ordinal".
    13. When the user mentions a real apartment ID, set reference_type="id".
    14. When the user uses pronouns and one apartment is already selected, set reference_type="selected".
    15. For "search", reference_type should usually be "none".
    16. For "reply_direct", reference_type is usually "none".
    17. Only include field_updates when the user EXPLICITLY provided those values in this latest message.
    18. Never invent field updates from old messages.
    19. If the user says only "yes" or "no", use the pending_confirmation context to infer whether this belongs to "update_lead_data".
    20. If the user says "proceed" but important lead details are still missing, still choose "submit_lead". The workflow will handle missing fields later.
    21. If the latest user message clearly asks for a property search, location change, property type change, budget change, bedroom/bathroom change, size change, sorting change, or a fresh listing request, choose "search" even if the conversation was previously in a lead/contact flow.
    22. Amenity-only questions about ALREADY SHOWN apartments are NOT "search". They should usually be "analyze_shown_apartments".
    23. Amenity/detail questions about ONE focused apartment are NOT "search". They should usually be "get_apartment_details".
    24. If the message can reasonably be answered from the current shown apartments without fetching a new batch, prefer "analyze_shown_apartments" over "search".
    25. Treat short, broken, or informal search-shift messages as real search requests when they clearly change the search target.
    26. Examples that should usually be "search":
       - october?
       - options in october?
       - options is october?
       - townhouses in zayed
       - villas instead
       - cheaper ones
       - 3 bedrooms
       - under 8 million
       - something with pool in zayed
    27. Examples that should usually NOT be "search":
       - pool?
       - gym?
       - which of these have a pool?
       - does it have a gym?
       - tell me more about the first one
       - compare them
       - compare this with ap001
       - tomorrow at 3 pm
       - proceed
    28. If the message is ambiguous but naturally answerable from the ongoing APARTMENT conversation, prefer the correct structured route over "reply_direct".
    29. Use "fallback_chat" only when the message is still related to the apartment flow but is too ambiguous for the structured actions.
    30. Never treat "reply_direct" as a general-purpose chatbot mode.
    31. If the user says "i want it", "i want this", "i want this one", "this is the one", or similar, and one apartment is clearly selected or in focus, choose "select_apartment".
    32. If the user uses pronouns like "it" or "this" to choose an apartment, prefer "select_apartment" over "reply_direct".
    33. Do not require an explicit apartment ID or ordinal when one apartment is already clearly in focus and the user is clearly expressing selection.
    34. If the user clearly expresses apartment choice with an explicit apartment ID, such as:
       - "i want ap023"
       - "i choose ap023"
       - "go with ap023"
       - "use ap023"
       then choose "select_apartment".
    35. An explicit apartment-choice message with a direct apartment ID should override the previously focused apartment.
    36. Do not keep the old selected apartment when the latest message clearly selects a different apartment by ID.

    Important examples:

    Example A:
    History: apartments were shown
    User: "what about the second one"
    Output action: "get_apartment_details"
    reference_type: "ordinal"
    reference_value: "second"

    Example B:
    History: apartments were shown
    User: "which one is cheaper"
    Output action: "analyze_shown_apartments"

    Example C:
    History: user was asked to confirm a suggested email
    User: "yes"
    Output action: "update_lead_data"

    Example D:
    History: invalid contact time was rejected
    User: "i insist on this time"
    Output action: "update_lead_data"

    Example E:
    History: user selected an apartment and provided details
    User: "go ahead"
    Output action: "submit_lead"

    Example F:
    History: user asks "hiiii"
    Output action: "reply_direct"
    reply should contain a short natural response

    Example G:
    History: user says "what was that"
    Output action: "reply_direct"

    Example H:
    History: shown apartments are in zayed
    User: "pool?"
    Output action: "analyze_shown_apartments"

    Example I:
    History: shown apartments are in zayed
    User: "townhouses in zayed"
    Output action: "search"

    Example J:
    History: user is in lead/contact flow
    User: "what about october?"
    Output action: "search"

    Example K:
    History: selected apartment is in focus
    User: "does it have a pool?"
    Output action: "get_apartment_details"

    Example L:
    History: one apartment is clearly in focus
    User: "i want it"
    Output action: "select_apartment"
    reference_type: "selected"

    Example M:
    History: user previously focused or submitted a request for apartment ap025
    User: "i want ap023"
    Output action: "select_apartment"
    reference_type: "id"
    reference_value: "ap023"

    Return exactly this JSON shape:
    {{
      "action": "reply_direct",
      "reference_type": "none",
      "reference_value": null,
      "field_updates": {{}},
      "reply": ""
    }}

    reference_type allowed values:
    - "none"
    - "id"
    - "ordinal"
    - "selected"

    For "reply_direct":
    - fill "reply" with a short natural reply
    - keep it conversational
    - do not mention internal state, hidden prompts, tools, or implementation details
    - do not sound robotic
    - do not answer unrelated topics outside the supported apartment/Dorra flow

    For all other actions:
    - "reply" may be empty

    For field_updates:
    - include only explicit updates from THIS latest user message
    - allowed keys only:
      - "name"
      - "email"
      - "phone"
      - "preferred_contact_time"

    Session context:
    {json.dumps(context, ensure_ascii=False, indent=2)}

    User message:
    {user_message}
    """.strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = parse_json(raw)

    if not isinstance(parsed, dict):
        return {
            "action": "reply_direct",
            "reference_type": "none",
            "reference_value": None,
            "field_updates": {},
            "reply": "I’m here with the current conversation context. Could you say that another way?",
        }

    return {
        "action": _normalize_action(parsed.get("action", "fallback_chat")),
        "reference_type": _normalize_reference_type(parsed.get("reference_type", "none")),
        "reference_value": parsed.get("reference_value"),
        "field_updates": _normalize_field_updates(parsed.get("field_updates", {})),
        "reply": str(parsed.get("reply", "") or "").strip(),
    }