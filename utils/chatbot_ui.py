"""
utils/chatbot_ui.py
--------------------
Presentation layer ONLY for the AI Chatbot ("AI Data Mentor") panel.

This module contains NO analytics, NO evidence logic, and NO knowledge of
benchmark vs. generic datasets. It renders a conversation UI (message list,
quick-prompt chips, chat input) and calls a `respond_fn(question) -> str`
callback supplied by app.py for every message - the SAME callback used for
both quick prompts and typed questions, so there is exactly one processing
path, not a separate "fake button" path.

UI/UX pattern (container `key=` -> a stable `.st-key-<key>` CSS class,
`st.chat_message`/`st.chat_input`, quick-prompt chip grid, fixed-height
auto-scrolling message list) is adapted from the supplied chatbot.py
reference file's visual direction. Its actual backend
(`backend.openrouter_client.generate_chat_response`, a "Career Mentor"
persona) does not exist in this project and is NOT imported here - all
responses come from the `respond_fn` this module is given.

Floating position (bottom-right via CSS `position: fixed` scoped to the
`.st-key-lm_ai_chatbot` container class) is a best-effort progressive
enhancement, in the same spirit as the sidebar nav-group CSS in app.py: if a
future Streamlit DOM/version doesn't apply it, the panel simply renders
in-line at the bottom of the page instead of floating - it cannot overlap
or break anything either way, and this was not visually verified in a
browser in this environment.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

ASSISTANT_AVATAR = "\U0001F9E0"  # matches the AI Insight Copilot's own icon
USER_AVATAR = "\U0001F64B"

MESSAGE_LIST_HEIGHT = 320
CHAT_STATE_KEY = "chat_messages"

_CHAT_CSS = """
<style>
/* Best-effort: pin the chatbot panel to the bottom-right corner. If this
   selector doesn't match the installed Streamlit version's DOM, the panel
   below simply stays in normal in-line document flow instead - it never
   disappears and never overlaps anything either way. */
.st-key-lm_ai_chatbot {
    position: fixed;
    right: 1.25rem;
    bottom: 1.25rem;
    width: min(380px, 92vw);
    max-height: 82vh;
    overflow-y: auto;
    z-index: 999;
    background: var(--lm-card, #102019);
    border: 1px solid var(--lm-border-green, rgba(34,197,94,0.35));
    border-radius: 16px;
    padding: 0.9rem 1rem 0.6rem 1rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.45);
}
.st-key-lm_ai_chatbot [data-testid="stChatMessage"] {
    padding: 6px 10px;
    border-radius: 12px;
    margin-bottom: 4px;
}
.st-key-lm_ai_chatbot .stButton > button {
    border-radius: 16px;
    padding: 4px 10px;
    font-size: 0.8rem;
    text-align: left;
    white-space: normal;
    line-height: 1.25;
}
.st-key-lm_ai_chatbot h4 { margin-bottom: 0.1rem; }
.st-key-lm_ai_chatbot [data-testid="stCaptionContainer"] { margin-bottom: 0.4rem; }

/* Narrow screens: don't let the panel eat the whole viewport width/height */
@media (max-width: 480px) {
    .st-key-lm_ai_chatbot {
        width: min(340px, 94vw);
        max-height: 70vh;
        padding: 0.7rem 0.8rem 0.5rem 0.8rem;
    }
    .st-key-lm_ai_chatbot [data-testid="stChatMessage"] { font-size: 0.88rem; }
    .st-key-lm_ai_chatbot .stButton > button { font-size: 0.75rem; padding: 4px 8px; }
}
</style>
"""

WELCOME_MESSAGE = (
    "\U0001F44B **Hi, I'm your AI Data Mentor.**\n\n"
    "Ask me anything about the currently active dataset - dataset findings, "
    "placement patterns, work-experience differences, model results, or "
    "student segments - or tap a quick prompt below to get started."
)


def _send(text: str, respond_fn: Callable[[str], str]) -> None:
    """Append the user's message, call the SAME respond_fn used for every
    message in this panel, and store the reply. No dataset/evidence logic
    lives here - respond_fn (owned by app.py) decides how to ground the
    answer in the active dataset."""
    text = text.strip()
    if not text:
        return

    history = st.session_state[CHAT_STATE_KEY]
    history.append({"role": "user", "content": text})

    with st.spinner("Thinking through the active dataset..."):
        reply = respond_fn(text)

    history.append({"role": "assistant", "content": reply})
    st.rerun()


def _render_message_list(history: list[dict[str, str]]) -> None:
    with st.container(height=MESSAGE_LIST_HEIGHT):
        if not history:
            st.markdown(WELCOME_MESSAGE)
        for turn in history:
            avatar = ASSISTANT_AVATAR if turn["role"] == "assistant" else USER_AVATAR
            with st.chat_message(turn["role"], avatar=avatar):
                st.markdown(turn["content"])


def _render_quick_prompts(quick_prompts: list[str], respond_fn: Callable[[str], str]) -> None:
    st.markdown("**Quick prompts:**")
    cols = st.columns(2)
    for i, prompt in enumerate(quick_prompts):
        with cols[i % 2]:
            if st.button(prompt, key=f"lm_chat_quick_{i}", use_container_width=True):
                _send(prompt, respond_fn)


def render_chatbot_panel(
    respond_fn: Callable[[str], str],
    quick_prompts: list[str],
    subtitle: str,
) -> None:
    """Render the AI Chatbot / AI Data Mentor panel.

    Args:
        respond_fn: Callable(question: str) -> str. The ONLY place where
            a question actually gets answered - owned by app.py, which
            decides whether to ground the answer in benchmark evidence or
            generic-dataset evidence. This module never calls analytics
            code directly.
        quick_prompts: the six fixed quick-prompt strings, sent through
            respond_fn exactly like typed text - not a separate hard-coded
            response path.
        subtitle: one line describing the active dataset, shown under the
            panel header.
    """
    st.session_state.setdefault(CHAT_STATE_KEY, [])
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    with st.container(key="lm_ai_chatbot"):
        st.markdown(
            '<span class="lm-badge">AI DATA MENTOR</span>', unsafe_allow_html=True
        )
        st.markdown("#### \U0001F916 AI Chatbot")
        st.caption(subtitle)

        history = st.session_state[CHAT_STATE_KEY]
        _render_message_list(history)

        # Quick prompts only shown before the conversation starts - keeps
        # the panel compact once the user is actually chatting.
        if not history:
            _render_quick_prompts(quick_prompts, respond_fn)
        else:
            if st.button("\U0001F5D1\uFE0F Clear conversation", key="lm_chat_clear"):
                st.session_state[CHAT_STATE_KEY] = []
                st.rerun()

        prompt_text = st.chat_input(
            "Ask something about your dataset...", key="lm_chat_input"
        )
        if prompt_text:
            _send(prompt_text, respond_fn)
