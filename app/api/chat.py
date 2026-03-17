from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.workflow import chat_graph, stream_graph

router = APIRouter(prefix="/chat", tags=["chat"])
session_mem = {}


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


@router.post("")
async def chat(payload: ChatRequest):
    session_id = payload.session_id.strip()
    message = payload.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    previous_state = session_mem.get(session_id, {})
    input_state = {**previous_state, "user_query": message}
    result = chat_graph.invoke(input_state)
    session_mem[session_id] = result

    return {
        "reply": result.get("reply", ""),
        "session_id": session_id,
    }


@router.post("/stream")
async def chat_stream(payload: ChatRequest):
    session_id = payload.session_id.strip()
    message = payload.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    previous_state = session_mem.get(session_id, {})

    def generate():
        latest_state = previous_state

        for chunk, new_state in stream_graph(message, previous_state):
            latest_state = new_state
            yield chunk

        session_mem[session_id] = latest_state

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "connection": "keep-alive", "X-Accel-Buffering": "no"},
    )