import json

import httpx
import streamlit as st

BACKEND_URL = "http://localhost:8000/api/chat"

st.set_page_config(page_title="Arxiv Research Agent", layout="centered")
st.title("Arxiv Research Agent")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
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

    # Stream response from backend
    with st.chat_message("assistant"):
        events_container = st.expander("Agent thought process", expanded=True)
        answer_placeholder = st.empty()

        collected_events: list[dict] = []
        answer_parts: list[str] = []

        try:
            with httpx.stream(
                "POST",
                BACKEND_URL,
                json={"question": prompt},
                timeout=120.0,
            ) as response:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        break

                    event = json.loads(payload)
                    event_type = event["event_type"]
                    content = event["content"]

                    if event_type == "thought":
                        with events_container:
                            st.caption(f"**{content}**")
                        collected_events.append(event)

                    elif event_type == "tool_call":
                        with events_container:
                            st.code(content, language="text")
                        collected_events.append(event)

                    elif event_type == "tool_result":
                        with events_container:
                            st.text(content[:1500])
                        collected_events.append(event)

                    elif event_type == "answer":
                        answer_parts.append(content)
                        answer_placeholder.markdown("".join(answer_parts))

        except httpx.ConnectError:
            answer_parts.append(
                "Could not connect to backend. "
                "Make sure the FastAPI server is running on http://localhost:8000"
            )
            answer_placeholder.markdown("".join(answer_parts))

        final_answer = "".join(answer_parts)
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "events": collected_events,
        })
