from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
from app.graph.workflow import chat_graph

router = APIRouter(prefix="/chat", tags=["chat"])
session_store = {}


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


def to_sse_event(data: dict, event: str | None = None) -> str:
    message = ""
    if event:
        message += f"event: {event}\n"
    message += f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    return message


def append_history(history, role, content):
    history = list(history or [])
    text = str(content or "").strip()
    if not text:
        return history

    history.append(
        {
            "role": role,
            "content": text,
        }
    )
    return history


@router.post("/stream")
async def chat_stream(request_data: ChatRequest):
    session_id = request_data.session_id.strip()
    user_message = request_data.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    if not user_message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    saved_state = dict(session_store.get(session_id, {}) or {})
    existing_history = saved_state.get("chat_history", [])
    updated_history = append_history(existing_history, "user", user_message)

    # Explicitly preserve all important conversational state across turns
    chat_state = {
        **saved_state,
        "chat_history": updated_history,
        "user_query": user_message,
        "pending_confirmation": saved_state.get("pending_confirmation", {}),
    }

    print("DEBUG chat_stream incoming session_id =", session_id)
    print("DEBUG chat_stream saved pending_confirmation =", saved_state.get("pending_confirmation"))

    def event_generator():
        final_state = dict(chat_state)
        sent_text = ""

        try:
            yield to_sse_event({"status": "started"}, event="start")

            for stream_type, stream_data in chat_graph.stream(
                chat_state,
                stream_mode=["updates", "custom"],
            ):
                if stream_type == "custom":
                    if stream_data:
                        text = str(stream_data)
                        if text:
                            sent_text += text
                            yield to_sse_event({"token": text}, event="token")
                    continue

                if stream_type != "updates":
                    continue

                for node_name, state_change in stream_data.items():
                    if not isinstance(state_change, dict):
                        continue

                    print(f"DEBUG node={node_name} state_change keys={list(state_change.keys())}")
                    if "pending_confirmation" in state_change:
                        print(f"DEBUG node={node_name} pending_confirmation={state_change.get('pending_confirmation')}")

                    final_state.update(state_change)

                    message_text = state_change.get("stream_text") or state_change.get("reply")
                    if not message_text:
                        continue

                    message_text = str(message_text)

                    if message_text.startswith(sent_text):
                        delta = message_text[len(sent_text):]
                    else:
                        delta = message_text

                    if delta:
                        sent_text += delta
                        yield to_sse_event({"token": delta}, event="token")

            yield to_sse_event({"status": "done"}, event="done")

        except Exception as exc:
            yield to_sse_event({"error": str(exc)}, event="error")

        finally:
            assistant_reply = str(final_state.get("reply") or sent_text).strip()
            final_history = final_state.get("chat_history", updated_history)
            final_history = append_history(final_history, "assistant", assistant_reply)
            final_state["chat_history"] = final_history

            # Keep pending_confirmation explicit so it cannot be lost silently
            final_state["pending_confirmation"] = final_state.get("pending_confirmation", {})

            print("DEBUG chat_stream final pending_confirmation =", final_state.get("pending_confirmation"))

            session_store[session_id] = final_state

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )