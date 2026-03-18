from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from app.graph.state import ChatState
from app.services.detect_intent import detect_intent
from app.services.lead_prepare import (build_success_reply,build_missing_reply,extract_lead_info,get_missing_fields,merge_lead_data)
from app.services.llm_chatbot import (extract_meta,search_apartments,build_final_output,render_reply,company_info_stream_to_writer)
from app.services.email_gen import send_email


def intent_node(state):
    user_message = state["user_query"]
    raw_result = detect_intent(user_message)

    if isinstance(raw_result, dict):
        intent = str(raw_result.get("intent", "search")).strip().lower()
    elif isinstance(raw_result, str):
        intent = raw_result.strip().lower()
        raw_result = {"intent": intent}
    else:
        intent = "search"
        raw_result = {"intent": "search"}

    if intent not in {"search", "lead", "company_info"}:
        intent = "unsupported"
        raw_result = {"intent": "unsupported"}

    return {
        "intent_result": raw_result,
        "intent": intent,
    }


def search_node(state):
    user_message = state["user_query"]

    filters = extract_meta(user_message)
    matches = search_apartments(user_message, filters, 15)
    final_output = build_final_output(user_message, matches,filters)
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
def company_info_node(state):
    user_message = state["user_query"]
    reply = company_info_stream_to_writer(user_message)

    return {
        "reply": reply,
        "stream_text": reply,
    }


def lead_node(state):
    user_message = state["user_query"]
    current_lead = state.get("lead_data", {})

    new_lead = extract_lead_info(user_message)
    merged_lead = merge_lead_data(current_lead, new_lead)
    missing_fields = get_missing_fields(merged_lead)

    return {
        "lead_data": merged_lead,
        "missing_fields": missing_fields,
    }


def missing_lead_info_node(state):
    lead_data = state.get("lead_data", {})
    missing_fields = state.get("missing_fields", [])

    reply = build_missing_reply(lead_data, missing_fields)

    return {
        "reply": reply,
        "stream_text": reply,
    }


def send_lead_node(state):
    lead_data = state.get("lead_data", {})
    matches = state.get("matches", [])

    apartment_id = str(lead_data.get("apartment_id", "")).lower()
    selected_apartment = None

    for match in matches:
        if str(match.get("apartment_id", "")).lower() == apartment_id:
            selected_apartment = match
            break

    if selected_apartment is None:
        reply = (
            f"I have all the lead details, but I could not find apartment {apartment_id} "
            f"in the current session results. Please search for it again, then resend your request."
        )
        return {
            "reply": reply,
            "stream_text": reply,
        }

    result = send_email(selected_apartment, lead_data)

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

def unsupported_node(state):
    reply = (
        """I can help with apartment search, lead requests, or Dorra company information only.\nTry asking for a property, sharing your contact details for a property, or asking about Dorra.\nOr you can contact one of our sales team via Hotline: 16077 or Email: info@dorra.com
        """
    )
    return {
        "reply": reply,
        "stream_text": reply,
    }

def build_chat_graph():
    graph = StateGraph(ChatState)

    graph.add_node("detect_intent", intent_node)
    graph.add_node("search_and_recommend", search_node)
    graph.add_node("company_info", company_info_node)
    graph.add_node("extract_lead", lead_node)
    graph.add_node("lead_reply", missing_lead_info_node)
    graph.add_node("send_lead", send_lead_node)
    graph.add_node("unsupported", unsupported_node)
    graph.add_edge(START, "detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        lambda state: state.get("intent", "unsupported"),
        {
            "search": "search_and_recommend",
            "lead": "extract_lead",
            "company_info": "company_info",
            "unsupported": "unsupported"
        },
    )

    graph.add_edge("search_and_recommend", END)
    graph.add_edge("company_info", END)
    graph.add_edge("unsupported", END)

    graph.add_conditional_edges(
        "extract_lead",
        lambda state: "missing" if state.get("missing_fields", []) else "complete",
        {
            "missing": "lead_reply",
            "complete": "send_lead",
        },
    )

    graph.add_edge("lead_reply", END)
    graph.add_edge("send_lead", END)

    return graph.compile()


chat_graph = build_chat_graph()