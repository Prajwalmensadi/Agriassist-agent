"""Streamlit chat UI for the local, evidence-grounded AgriAssist MVP."""

from __future__ import annotations

import streamlit as st

from agriassist.controller import ConversationState, handle_message


st.set_page_config(page_title="AgriAssist", page_icon="🌾", layout="centered")
st.title("AgriAssist")
st.caption("Paddy (rice) information assistant · Coastal Karnataka MVP")
st.info(
    "This MVP supports paddy/rice only. Its documents are provisional and may not be "
    "locally validated. AgriAssist is not a certified agricultural adviser."
)

with st.sidebar:
    st.subheader("About this prototype")
    st.write(
        "Answers use passages from a small local document set. Source location and "
        "verification limits are shown with each answer. Retrieval similarity does "
        "not prove that a statement is correct."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.controller_state = ConversationState()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "controller_state" not in st.session_state:
    st.session_state.controller_state = ConversationState()


def _render_sources(evidence: list[dict]) -> None:
    if not evidence:
        return
    with st.expander("Sources and limitations"):
        for item in evidence:
            filename = item.get("source_filename", "Source filename unavailable")
            page = item.get("page_number")
            location = f"page {page}" if isinstance(page, int) else "page unavailable"
            metadata = item.get("metadata", {})
            title = metadata.get("source_title")
            heading = f"{title} · {filename} · {location}" if title else f"{filename} · {location}"
            st.markdown(f"**{heading}**")
            st.caption(f"Geographic applicability: {metadata.get('geographic_applicability', 'unknown')}")
            st.caption(f"Verification: {metadata.get('verification_status', 'unknown')}")
            publisher = metadata.get("publisher")
            if publisher:
                st.caption(f"Publisher stated in inventory: {publisher}")
            source_url = metadata.get("original_source_url")
            if source_url:
                st.caption(f"Source URL stated in document: {source_url}")


def _evidence_for_history(result) -> list[dict]:
    return [
        {
            "source_filename": item.source_filename,
            "page_number": item.page_number,
            "metadata": item.metadata,
        }
        for item in result.evidence
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("evidence", []))

question = st.chat_input("Ask a question about paddy (rice)")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking paddy documents and preparing a response…"):
            try:
                result = handle_message(
                    question,
                    conversation=st.session_state.controller_state,
                    top_k=4,
                )
                st.session_state.controller_state = result.conversation
                evidence = _evidence_for_history(result)
                st.markdown(result.answer)
                _render_sources(evidence)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer, "evidence": evidence}
                )
            except Exception:
                st.error(
                    "AgriAssist could not complete that request. Please try again later or "
                    "contact a local agricultural extension officer for urgent guidance."
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "AgriAssist could not complete that request. Please try again later.",
                        "evidence": [],
                    }
                )
