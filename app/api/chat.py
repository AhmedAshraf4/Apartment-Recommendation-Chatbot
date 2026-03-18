from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import time
from app.graph.workflow import chat_graph

router = APIRouter(prefix="/chat", tags=["chat"])
session_store = {}


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


def split_for_typing_effect(text, size=18):
    text = text or ""
    for i in range(0, len(text), size):
        yield text[i:i + size]


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

    def stream_response():
        final_state = dict(chat_state)

        try:
            for stream_type, stream_data in chat_graph.stream(
                chat_state,
                stream_mode=["updates", "custom"],
            ):
                if stream_type == "custom":
                    if stream_data:
                        yield stream_data
                    continue

                if stream_type != "updates":
                    continue

                for _, state_change in stream_data.items():
                    if not isinstance(state_change, dict):
                        continue

                    final_state.update(state_change)

                    message_text = state_change.get("stream_text") or state_change.get("reply")
                    current_intent = final_state.get("intent")

                    if message_text and current_intent != "company_info":
                        for piece in split_for_typing_effect(message_text, 8):
                            yield piece
                            time.sleep(0.04)
        finally:
            session_store[session_id] = final_state

    return StreamingResponse(
        stream_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )