from fastapi import APIRouter
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
    state = session_mem.get(session_id, {})
    state = {**state, "user_query": message}
    result = chat_graph.invoke(state)
    session_mem[session_id] = result

    return {
        "reply": result.get("reply", ""),
        "session_id": session_id,
    }


@router.post("/stream")
async def chat_stream(payload: ChatRequest):
    session_id = payload.session_id.strip()
    message = payload.message.strip()
    state = session_mem.get(session_id, {})

    def generate():
        current_state = state
        for chunk, new_state in stream_graph(message, state):
            current_state = new_state
            yield chunk

        session_mem[session_id] = current_state

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )