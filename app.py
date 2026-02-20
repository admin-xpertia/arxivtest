import asyncio
import json

import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = "mistral-medium-latest"

SYSTEM_PROMPT = (
    "You are a helpful scientific research assistant. "
    "Use the Arxiv tool to search for academic papers when needed. "
    "Always cite paper titles and authors when referencing results. "
    "Answer in the same language the user writes in."
)

# ---------------------------------------------------------------------------
# Agent (cached so it's built once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def build_agent() -> AgentExecutor:
    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=0,
    )
    tools = [ArxivQueryRun()]
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


# ---------------------------------------------------------------------------
# Async runner – collects events from astream_events
# ---------------------------------------------------------------------------

async def _collect_events(executor: AgentExecutor, question: str) -> list[dict]:
    """Run the agent and return a flat list of events."""
    events: list[dict] = []
    async for event in executor.astream_events({"input": question}, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_start":
            events.append({"event_type": "thought", "content": "Thinking..."})

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    if tc.get("name"):
                        events.append({
                            "event_type": "tool_call",
                            "content": f"Calling tool: {tc['name']}",
                        })
                    if tc.get("args"):
                        events.append({"event_type": "tool_call", "content": tc["args"]})
            elif chunk.content:
                events.append({"event_type": "answer", "content": chunk.content})

        elif kind == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = event["data"].get("input", "")
            if isinstance(tool_input, dict):
                tool_input = json.dumps(tool_input, ensure_ascii=False)
            events.append({
                "event_type": "tool_call",
                "content": f"[{tool_name}] query: {tool_input}",
            })

        elif kind == "on_tool_end":
            output = event["data"].get("output", "")
            if hasattr(output, "content"):
                output = output.content
            output_str = str(output)
            if len(output_str) > 2000:
                output_str = output_str[:2000] + "..."
            events.append({"event_type": "tool_result", "content": output_str})

    return events


def run_agent(question: str) -> list[dict]:
    executor = build_agent()
    return asyncio.run(_collect_events(executor, question))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Arxiv Research Agent", layout="centered")
st.title("Arxiv Research Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("events"):
            with st.expander("Agent thought process"):
                for ev in msg["events"]:
                    _type = ev["event_type"]
                    if _type == "thought":
                        st.caption(f"**{ev['content']}**")
                    elif _type == "tool_call":
                        st.code(ev["content"], language="text")
                    elif _type == "tool_result":
                        st.text(ev["content"][:1500])

# User input
if prompt := st.chat_input("Ask about scientific papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent and display results
    with st.chat_message("assistant"):
        with st.spinner("Searching papers and reasoning..."):
            events = run_agent(prompt)

        # Separate events
        thought_events = [e for e in events if e["event_type"] != "answer"]
        answer_parts = [e["content"] for e in events if e["event_type"] == "answer"]
        final_answer = "".join(answer_parts) or "I couldn't generate a response."

        # Show thought process
        if thought_events:
            with st.expander("Agent thought process", expanded=True):
                for ev in thought_events:
                    _type = ev["event_type"]
                    if _type == "thought":
                        st.caption(f"**{ev['content']}**")
                    elif _type == "tool_call":
                        st.code(ev["content"], language="text")
                    elif _type == "tool_result":
                        st.text(ev["content"][:1500])

        # Show final answer
        st.markdown(final_answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "events": thought_events,
        })
