import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from application.agent import run_agent_stream
from domain.models import QueryRequest

router = APIRouter()


@router.post("/chat")
async def chat(request: QueryRequest):
    """SSE endpoint that streams agent events as newline-delimited JSON."""

    async def event_generator():
        async for event in run_agent_stream(request.question):
            data = json.dumps(event.model_dump(), ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
