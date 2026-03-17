from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import time
from app.graph.workflow import chat_graph

router = APIRouter(prefix="/chat", tags=["chat"])
session_mem = {}


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


@router.post("/stream")
async def chat_stream(payload: ChatRequest):
    session_id = payload.session_id.strip()
    message = payload.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    previous_state = session_mem.get(session_id, {})
    input_state = {**previous_state, "user_query": message}

    def chunk_text_for_ui(text, chunk_size=18):
        text = text or ""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def generate():
        latest_state = dict(input_state)

        try:
            for mode, chunk in chat_graph.stream(
                    input_state,
                    stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    if chunk:
                        yield chunk

                elif mode == "updates":
                    for _, node_update in chunk.items():
                        if not isinstance(node_update, dict):
                            continue

                        latest_state.update(node_update)

                        stream_text = node_update.get("stream_text") or node_update.get("reply")
                        intent = latest_state.get("intent")

                        if stream_text and intent != "company_info":
                            for piece in chunk_text_for_ui(stream_text, 18):
                                yield piece
                                time.sleep(0.03)
        finally:
            session_mem[session_id] = latest_state
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )