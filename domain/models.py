from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class AgentEvent(BaseModel):
    """A single event emitted during agent execution."""
    event_type: str  # "thought", "tool_call", "tool_result", "answer"
    content: str
