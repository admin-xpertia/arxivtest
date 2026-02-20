import json
from collections.abc import AsyncGenerator

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from domain.models import AgentEvent
from infrastructure.llm import get_llm
from infrastructure.tools import get_arxiv_tool

SYSTEM_PROMPT = (
    "You are a helpful scientific research assistant. "
    "Use the Arxiv tool to search for academic papers when needed. "
    "Always cite paper titles and authors when referencing results. "
    "Answer in the same language the user writes in."
)


def _build_agent() -> AgentExecutor:
    llm = get_llm()
    tools = [get_arxiv_tool()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
    )


_executor = _build_agent()


async def run_agent_stream(question: str) -> AsyncGenerator[AgentEvent, None]:
    """Stream agent events as they happen using astream_events."""
    async for event in _executor.astream_events(
        {"input": question},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_start":
            yield AgentEvent(event_type="thought", content="Thinking...")

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # If the model is producing tool calls, report them
            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    if tc.get("name"):
                        yield AgentEvent(
                            event_type="tool_call",
                            content=f"Calling tool: {tc['name']}",
                        )
                    if tc.get("args"):
                        yield AgentEvent(
                            event_type="tool_call",
                            content=tc["args"],
                        )
            # If the model is producing text (final answer)
            elif chunk.content:
                yield AgentEvent(event_type="answer", content=chunk.content)

        elif kind == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = event["data"].get("input", "")
            if isinstance(tool_input, dict):
                tool_input = json.dumps(tool_input, ensure_ascii=False)
            yield AgentEvent(
                event_type="tool_call",
                content=f"[{tool_name}] query: {tool_input}",
            )

        elif kind == "on_tool_end":
            output = event["data"].get("output", "")
            if hasattr(output, "content"):
                output = output.content
            output_str = str(output)
            # Truncate very long tool outputs for the stream
            if len(output_str) > 2000:
                output_str = output_str[:2000] + "..."
            yield AgentEvent(
                event_type="tool_result",
                content=output_str,
            )
