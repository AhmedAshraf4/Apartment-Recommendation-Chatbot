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


@router.post("/stream")
async def chat_stream(request_data: ChatRequest):
    session_id = request_data.session_id.strip()
    user_message = request_data.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    if not user_message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    saved_state = session_store.get(session_id, {})
    chat_state = {**saved_state, "user_query": user_message}

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

                for _, state_change in stream_data.items():
                    if not isinstance(state_change, dict):
                        continue

                    final_state.update(state_change)

                    message_text = state_change.get("stream_text") or state_change.get("reply")
                    if not message_text:
                        continue

                    message_text = str(message_text)

                    # Only send the new delta, not the full text again
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