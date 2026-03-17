from typing import TypedDict

class ChatState(TypedDict, total=False):
    user_query: str
    intent: str
    intent_result: dict
    filters: dict
    matches: list
    lead_data: dict
    missing_fields: list[str]
    reply: str
    recommendations: list
    company_note: str