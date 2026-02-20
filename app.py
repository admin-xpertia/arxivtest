import streamlit as st
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent

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
def build_agent():
    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=0,
    )
    tools = [ArxivQueryRun()]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def run_agent(question: str) -> list[dict]:
    """Run the agent and collect events (thoughts, tool calls, answer)."""
    agent = build_agent()
    events: list[dict] = []

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        # Each chunk is a dict with the node name as key
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                # AI message with tool calls → agent is deciding to use a tool
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        events.append({
                            "event_type": "tool_call",
                            "content": f"Calling tool: {tc['name']}\nArgs: {tc['args']}",
                        })

                # Tool message → result from tool execution
                elif msg.type == "tool":
                    output_str = str(msg.content)
                    if len(output_str) > 2000:
                        output_str = output_str[:2000] + "..."
                    events.append({
                        "event_type": "tool_result",
                        "content": output_str,
                    })

                # AI message without tool calls → final answer
                elif msg.type == "ai" and msg.content:
                    events.append({
                        "event_type": "answer",
                        "content": msg.content,
                    })

    return events


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
                    if _type == "tool_call":
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
                    if _type == "tool_call":
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
