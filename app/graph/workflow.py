from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.graph.state import ChatState
from app.services.detect_intent import detect_intent
from app.services.lead_prepare import (
    build_success_reply,
    build_missing_reply,
    extract_lead_info,
    get_missing_fields,
    merge_lead_data,
)
from app.services.llm_chatbot import (
    extract_meta,
    search_apartments,
    build_final_output,
    render_reply,
    company_info_stream_to_writer,
)
from app.services.email_gen import send_email


def normalize_intent_result(intent_result):
    if isinstance(intent_result, dict):
        intent = str(intent_result.get("intent", "search")).strip().lower()
    elif isinstance(intent_result, str):
        intent = intent_result.strip().lower()
        intent_result = {"intent": intent}
    else:
        intent = "search"
        intent_result = {"intent": "search"}

    if intent not in {"search", "lead", "company_info"}:
        intent = "search"
        intent_result = {"intent": "search"}

    return {
        "intent_result": intent_result,
        "intent": intent,
    }


def detect_intent_node(state: ChatState):
    user_query = state["user_query"]
    intent_result = detect_intent(user_query)
    return normalize_intent_result(intent_result)


def search_node(state: ChatState):
    user_query = state["user_query"]

    filters = extract_meta(user_query)
    matches = search_apartments(user_query, filters, 15)
    final_output = build_final_output(user_query, matches)
    reply = render_reply(final_output)

    return {
        "filters": filters,
        "matches": matches,
        "recommendations": final_output.get("recommendations", []),
        "company_note": final_output.get("company_note", ""),
        "reply": reply,
        "stream_text": reply,
    }


@traceable(name="company_info_node")
def company_info_node(state: ChatState):
    user_query = state["user_query"]
    reply = company_info_stream_to_writer(user_query)

    return {
        "reply": reply,
        "stream_text": reply,
    }


def extract_lead_node(state: ChatState):
    user_query = state["user_query"]
    existing_lead_data = state.get("lead_data", {})

    new_lead_data = extract_lead_info(user_query)
    merged_lead_data = merge_lead_data(existing_lead_data, new_lead_data)
    missing_fields = get_missing_fields(merged_lead_data)

    return {
        "lead_data": merged_lead_data,
        "missing_fields": missing_fields,
    }


def lead_reply_node(state: ChatState):
    lead_data = state.get("lead_data", {})
    missing_fields = state.get("missing_fields", [])

    reply = build_missing_reply(lead_data, missing_fields)

    return {
        "reply": reply,
        "stream_text": reply,
    }


def send_lead_node(state: ChatState):
    lead_data = state.get("lead_data", {})
    matches = state.get("matches", [])

    apartment_id = str(lead_data.get("apartment_id", "")).lower()
    apartment = None

    for match in matches:
        if str(match.get("apartment_id", "")).lower() == apartment_id:
            apartment = match
            break

    if apartment is None:
        reply = (
            f"I have all the lead details, but I could not find apartment {apartment_id} "
            f"in the current session results. Please search for it again, then resend your request."
        )
        return {
            "reply": reply,
            "stream_text": reply,
        }

    result = send_email(apartment, lead_data)

    if result.get("success"):
        reply = build_success_reply(lead_data)
        return {
            "reply": reply,
            "stream_text": reply,
            "lead_data": {},
            "missing_fields": [],
        }

    reply = result.get("message", "Failed to send email.")
    return {
        "reply": reply,
        "stream_text": reply,
    }


def route_intent(state: ChatState):
    return state.get("intent", "search")


def route_lead_completion(state: ChatState):
    missing_fields = state.get("missing_fields", [])
    return "missing" if missing_fields else "complete"


def build_chat_graph():
    graph = StateGraph(ChatState)

    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("search_and_recommend", search_node)
    graph.add_node("company_info", company_info_node)
    graph.add_node("extract_lead", extract_lead_node)
    graph.add_node("lead_reply", lead_reply_node)
    graph.add_node("send_lead", send_lead_node)

    graph.add_edge(START, "detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            "search": "search_and_recommend",
            "lead": "extract_lead",
            "company_info": "company_info",
        },
    )

    graph.add_edge("search_and_recommend", END)
    graph.add_edge("company_info", END)

    graph.add_conditional_edges(
        "extract_lead",
        route_lead_completion,
        {
            "missing": "lead_reply",
            "complete": "send_lead",
        },
    )

    graph.add_edge("lead_reply", END)
    graph.add_edge("send_lead", END)

    return graph.compile()


chat_graph = build_chat_graph()
