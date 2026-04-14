from typing import TypedDict

class ChatState(TypedDict, total=False):
    user_query: str

    # conversational memory
    chat_history: list[dict]
    conversation_summary: str

    # routing
    intent: str
    intent_result: dict
    action: str

    # search memory
    filters: dict
    last_search_filters: dict
    matches: list
    last_shown_apartments: list
    apartment_reference_map: dict

    # selected apartment memory
    selected_apartment_id: str
    selected_apartment: dict

    # user / lead draft memory
    user_profile: dict
    lead_data: dict
    missing_fields: list[str]

    # response
    reply: str
    stream_text: str