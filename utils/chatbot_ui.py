"""
utils/chatbot_ui.py
--------------------
Presentation layer ONLY for the AI Data Mentor chatbot panel.

This module contains NO analytics, NO evidence logic, and NO knowledge of
benchmark vs. generic datasets. It renders a conversation UI (message list,
quick-prompt chips, chat input) and calls a `respond_fn(question) -> str`
callback supplied by app.py for every message - the SAME callback used for
both quick prompts and typed questions, so there is exactly one processing
path, not a separate "fake button" path. respond_fn (`chatbot_respond()` in
app.py) and the quick-prompt intent mapping it uses are untouched by this
module.

Two independent session_state booleans control visibility:
- chatbot_enabled (owned by app.py's sidebar toggle): whether the feature
  exists at all this session. This module doesn't set it, only reads it
  implicitly (app.py skips calling render_chatbot_panel() entirely when
  it's False - see the call site at the bottom of app.py).
- chatbot_open (owned entirely by this module): whether the panel is
  currently expanded, vs. collapsed to a small floating launcher. Defaults
  to False (closed) via `setdefault`, so it is never reset to False on a
  rerun it already set to True - it only starts False once, the first time
  this module ever runs in a session.

UI/UX pattern (container `key=` -> a stable `.st-key-<key>` CSS class,
`st.chat_message`/`st.chat_input`, quick-prompt chip grid, fixed-height
auto-scrolling message list, and the launcher/open/close split itself) is
adapted from the supplied chatbot.py reference file's BEHAVIORAL pattern
only. Its actual backend (`backend.openrouter_client.generate_chat_response`,
a "Career Mentor" persona) does not exist in this project and is NOT
imported here - all responses come from the `respond_fn` this module is
given, and all branding/copy below is LearnMate Analytics AI's own
("AI Data Mentor / Powered by LearnMate Analytics AI"), not the reference
file's career-mentor content.

Floating position (bottom-right via CSS `position: fixed` scoped to a
`.st-key-<container-key>` class) is a best-effort progressive enhancement,
in the same spirit as the sidebar nav-group CSS in app.py: if a future
Streamlit DOM/version doesn't apply it, the launcher/panel simply renders
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
OPEN_STATE_KEY = "chatbot_open"

# Purely cosmetic: nicer emoji-decorated labels for the six fixed quick
# prompts. The KEY (plain text, no emoji) is what actually gets sent to
# respond_fn - unchanged from app.py's CHATBOT_QUICK_PROMPTS list, so its
# quick-prompt -> intent mapping keeps matching exactly as before. Only the
# button's visible text is decorated; a prompt not in this dict (there
# shouldn't be one) just displays as-is.
_QUICK_PROMPT_DISPLAY_LABELS = {
    "Summarize the main dataset findings": "\U0001F4CA Summarize the main dataset findings",
    "Explain the placement patterns": "\U0001F3AF Explain the placement patterns",
    "Analyze work-experience differences": "\U0001F4BC Analyze work-experience differences",
    "Explain the model results": "\U0001F916 Explain the model results",
    "Explain the student segments": "\U0001F465 Explain the student segments",
    "Help me interpret this dataset": "\U0001F4A1 Help me interpret this dataset",
}

_CHAT_CSS = """
<style>
/* Best-effort: pin the expanded chatbot panel to the bottom-right corner.
   If this selector doesn't match the installed Streamlit version's DOM,
   the panel below simply stays in normal in-line document flow instead -
   it never disappears and never overlaps anything either way. */
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

/* The "Close AI Data Mentor" button gets a plain, low-emphasis treatment
   (not the emerald gradient) since it's a dismiss action, not a CTA. */
.st-key-lm_ai_chatbot div[data-testid="stButton"]:first-of-type > button {
    text-align: center; font-weight: 600; color: var(--lm-text-secondary, #A7B8AE);
    background: transparent; border: 1px solid var(--lm-border, rgba(255,255,255,0.08));
}

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

_LAUNCHER_CSS = """
<style>
/* Best-effort: pin the collapsed launcher pill to the bottom-right corner,
   same fallback behavior as the panel above if unsupported. */
.st-key-lm_chatbot_launcher {
    position: fixed;
    right: 1.25rem;
    bottom: 1.25rem;
    width: auto;
    z-index: 1000;
}
.st-key-lm_chatbot_launcher .stButton > button {
    background: linear-gradient(135deg, var(--lm-green, #22C55E) 0%, var(--lm-emerald, #10B981) 100%);
    color: #06110B;
    font-weight: 700;
    border: none;
    border-radius: 999px;
    padding: 0.65rem 1.25rem;
    white-space: nowrap;
    box-shadow: 0 6px 20px rgba(16,185,129,0.40), 0 2px 6px rgba(0,0,0,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.st-key-lm_chatbot_launcher .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 26px rgba(16,185,129,0.50), 0 2px 8px rgba(0,0,0,0.4);
}

/* Mobile: keep it clear of the bottom edge / Streamlit's own bottom bar */
@media (max-width: 480px) {
    .st-key-lm_chatbot_launcher { right: 0.9rem; bottom: 1rem; }
    .st-key-lm_chatbot_launcher .stButton > button { padding: 0.55rem 1.05rem; font-size: 0.85rem; }
}
</style>
"""

WELCOME_MESSAGE = (
    "\U0001F44B **Hi, I'm your AI Data Mentor.**\n\n"
    "Ask me about the active dataset, placement patterns, work-experience "
    "differences, model results, student segments, salary insights, or "
    "other findings from your current analysis - or tap a quick prompt "
    "below to get started."
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
            label = _QUICK_PROMPT_DISPLAY_LABELS.get(prompt, prompt)
            if st.button(label, key=f"lm_chat_quick_{i}", use_container_width=True):
                _send(prompt, respond_fn)


def render_chatbot_panel(
    respond_fn: Callable[[str], str],
    quick_prompts: list[str],
    subtitle: str,
) -> None:
    """Render the AI Data Mentor chatbot: a small floating launcher when
    collapsed, or the full conversation panel when expanded.

    Args:
        respond_fn: Callable(question: str) -> str. The ONLY place where
            a question actually gets answered - owned by app.py, which
            decides whether to ground the answer in benchmark evidence or
            generic-dataset evidence. This module never calls analytics
            code directly.
        quick_prompts: the six fixed quick-prompt strings, sent through
            respond_fn exactly like typed text - not a separate hard-coded
            response path. Display labels are decorated with emoji here
            (see _QUICK_PROMPT_DISPLAY_LABELS); the text sent to
            respond_fn is unchanged.
        subtitle: one line describing the active dataset, shown under the
            panel header.
    """
    st.session_state.setdefault(CHAT_STATE_KEY, [])
    # setdefault only sets this ONCE per session (the first time this
    # function ever runs) - it never overwrites a True the user already
    # set by clicking the launcher, on this or any later rerun/page nav.
    st.session_state.setdefault(OPEN_STATE_KEY, False)

    if not st.session_state[OPEN_STATE_KEY]:
        st.markdown(_LAUNCHER_CSS, unsafe_allow_html=True)
        with st.container(key="lm_chatbot_launcher"):
            if st.button(
                "\U0001F4AC AI Data Mentor",
                key="lm_chatbot_launcher_button",
                use_container_width=True,
            ):
                st.session_state[OPEN_STATE_KEY] = True
                st.rerun()
        return

    st.markdown(_CHAT_CSS, unsafe_allow_html=True)
    with st.container(key="lm_ai_chatbot"):
        if st.button(
            "\u2715 Close AI Data Mentor", key="lm_chatbot_close", use_container_width=True
        ):
            st.session_state[OPEN_STATE_KEY] = False
            st.rerun()

        st.markdown("#### \U0001F9E0 AI Data Mentor")
        st.markdown(
            '<span class="lm-badge">Powered by LearnMate Analytics AI</span>',
            unsafe_allow_html=True,
        )
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
