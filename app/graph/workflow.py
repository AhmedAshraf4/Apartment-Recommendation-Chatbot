from langgraph.graph import END, START, StateGraph
from app.graph.state import ChatState
from app.services.detect_intent import detect_intent
from app.services.lead_prepare import (
    build_success_reply,
    build_missing_reply,
    extract_lead_info,
    get_missing_fields,
    merge_lead_data,
    stream_missing_reply,
    stream_success_reply,
    stream_error_reply,
    chunk_text,
)
from app.services.llm_chatbot import (
    extract_meta,
    search_apartments,
    generate_answer,
    validate_output,
    merge_recommendations,
    render_reply,
    stream_recommendation_text,
)
from app.services.email_gen import send_email
from langsmith import traceable


def detect_intent_node(state):
    user_query = state["user_query"]
    return {"intent": detect_intent(user_query)}


def search_node(state):
    user_query = state["user_query"]

    filters = extract_meta(user_query)
    matches = search_apartments(user_query, filters, 15)
    raw_output = generate_answer(user_query, matches)
    validated_output = validate_output(raw_output, matches)
    final_output = merge_recommendations(validated_output, matches)
    reply = render_reply(final_output)

    return {
        "filters": filters,
        "matches": matches,
        "reply": reply,
        "recommendations": final_output.get("recommendations", []),
        "company_note": final_output.get("company_note", ""),
    }


def get_lead_node(state):
    user_query = state["user_query"]
    existing_lead_data = state.get("lead_data", {})

    new_lead_data = extract_lead_info(user_query)
    merged_lead_data = merge_lead_data(existing_lead_data, new_lead_data)
    missing_fields = get_missing_fields(merged_lead_data)

    return {
        "lead_data": merged_lead_data,
        "missing_fields": missing_fields,
    }


def lead_reply_node(state):
    lead_data = state.get("lead_data", {})
    missing_fields = state.get("missing_fields", [])

    if missing_fields:
        reply = build_missing_reply(lead_data, missing_fields)
        return {"reply": reply}

    return {}


def send_lead_node(state):
    lead_data = state.get("lead_data", {})
    matches = state.get("matches", [])

    apartment_id = str(lead_data.get("apartment_id", "")).lower()
    apartment = None

    for match in matches:
        if str(match.get("apartment_id", "")).lower() == apartment_id:
            apartment = match
            break

    if apartment is None:
        return {
            "reply": (
                f"I have all the lead details, but I could not find apartment {apartment_id} "
                f"in the current session results. Please search for it again, then resend your request."
            )
        }

    result = send_email(apartment, lead_data)

    if result.get("success"):
        reply = build_success_reply(lead_data)
    else:
        reply = result.get("message", "Failed to send email.")

    return {"reply": reply}


def route_intent(state):
    return state.get("intent", "search")


def route_lead_completion(state):
    missing_fields = state.get("missing_fields", [])
    if missing_fields:
        return "missing"
    return "complete"


def build_chat_graph():
    graph = StateGraph(ChatState)

    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("search_and_recommend", search_node)
    graph.add_node("extract_lead", get_lead_node)
    graph.add_node("lead_reply", lead_reply_node)
    graph.add_node("send_lead", send_lead_node)

    graph.add_edge(START, "detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            "search": "search_and_recommend",
            "lead": "extract_lead",
        },
    )

    graph.add_edge("search_and_recommend", END)

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

@traceable(name="run_graph")
def run_graph(user_query, previous_state=None):
    previous_state = previous_state or {}
    input_state = {**previous_state, "user_query": user_query}
    return chat_graph.invoke(input_state)

@traceable(name="stream_graph")
def stream_graph(user_query, previous_state=None):
    previous_state = previous_state or {}

    intent = detect_intent(user_query)
    state = {**previous_state, "user_query": user_query, "intent": intent}

    # SEARCH FLOW
    if intent == "search":
        filters = extract_meta(user_query)
        matches = search_apartments(user_query, filters, 15)

        state["filters"] = filters
        state["matches"] = matches

        for chunk in stream_recommendation_text(user_query, matches):
            yield chunk, state
        return

    # LEAD FLOW
    if intent == "lead":
        existing_lead_data = state.get("lead_data", {})
        new_lead_data = extract_lead_info(user_query)
        merged_lead_data = merge_lead_data(existing_lead_data, new_lead_data)
        missing_fields = get_missing_fields(merged_lead_data)

        state["lead_data"] = merged_lead_data
        state["missing_fields"] = missing_fields

        if missing_fields:
            for chunk in stream_missing_reply(merged_lead_data, missing_fields):
                yield chunk, state
            return

        lead_data = state.get("lead_data", {})
        matches = state.get("matches", [])

        apartment_id = str(lead_data.get("apartment_id", "")).lower()
        apartment = None

        for match in matches:
            if str(match.get("apartment_id", "")).lower() == apartment_id:
                apartment = match
                break

        if apartment is None:
            error_message = (
                f"I have all the lead details, but I could not find apartment {apartment_id} "
                f"in the current session results. Please search for it again, then resend your request."
            )
            for chunk in stream_error_reply(error_message):
                yield chunk, state
            return

        result = send_email(apartment, lead_data)

        if result.get("success"):
            for chunk in stream_success_reply(lead_data):
                yield chunk, state
        else:
            for chunk in stream_error_reply(result.get("message", "Failed to send email.")):
                yield chunk, state
        return

    # FALLBACK
    fallback_message = "Sorry, I could not determine how to handle your request."
    for chunk in chunk_text(fallback_message):
        yield chunk, state