"""
Streamlit entry point for Neo-AI.

Provides a chat interface with optional document upload, concise/detailed mode,
and per-user conversation state. Run with:

    streamlit run main.py
"""
import os
import streamlit as st
from app.components.agent.agent import answer_query
from app.utils.process_doc.processing import process_document
from app.utils.summarize import summarize_document
from app.utils.logger.logger import get_logger
from app.exceptions import DocumentProcessingError

logger = get_logger()

# Session state keys and defaults (single source of truth)
SESSION_KEYS = (
    "vectorstore",
    "messages",
    "thread_id",
    "last_processed_file",
    "started",
    "current_document_name",
    "last_summary",
)
SESSION_DEFAULTS = {
    "vectorstore": None,
    "messages": [],
    "thread_id": "streamlit-default",
    "last_processed_file": None,
    "started": False,
    "current_document_name": None,
    "last_summary": None,
}


def document_display_name(filename: str) -> str:
    """Convert filename to a readable title for prompts (e.g. 'Summarize Attention is all you need')."""
    name = os.path.splitext(filename)[0]
    return name.replace("_", " ").replace("-", " ").strip()


class StreamlitFileAdapter:
    """
    Adapts Streamlit's UploadedFile to the interface expected by process_document.

    process_document expects a file-like object with .filename and .save(fp).
    Streamlit provides .name and .read(); this adapter bridges the two.
    """

    def __init__(self, uploaded_file):
        self.filename = getattr(uploaded_file, "name", "upload.txt")
        self._bytes = uploaded_file.read()

    def save(self, fp):
        fp.write(self._bytes)


def init_session_state() -> None:
    """Initialise Streamlit session state with default values if not already set."""
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = SESSION_DEFAULTS[key]


# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neo-AI", layout="wide", page_icon="⚡")
init_session_state()

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #09090f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8e8f0;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Hide default Streamlit chrome on landing ── */
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── LANDING PAGE ── */
.landing-wrap {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    padding: 2rem;
}

/* Animated gradient orbs */
.landing-wrap::before {
    content: '';
    position: fixed;
    top: -20%;
    left: -10%;
    width: 55vw;
    height: 55vw;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 65%);
    border-radius: 50%;
    animation: orb1 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
.landing-wrap::after {
    content: '';
    position: fixed;
    bottom: -15%;
    right: -5%;
    width: 45vw;
    height: 45vw;
    background: radial-gradient(circle, rgba(236,72,153,0.13) 0%, transparent 65%);
    border-radius: 50%;
    animation: orb2 15s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes orb1 { from { transform: translate(0,0) scale(1); } to { transform: translate(4vw, 6vh) scale(1.12); } }
@keyframes orb2 { from { transform: translate(0,0) scale(1); } to { transform: translate(-5vw, -4vh) scale(1.08); } }

/* Noise grain overlay */
.grain {
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 1;
    opacity: 0.5;
}

.landing-content {
    position: relative;
    z-index: 2;
    text-align: center;
    max-width: 820px;
    animation: fadeUp 0.9s cubic-bezier(.22,1,.36,1) both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(32px); }
    to   { opacity: 1; transform: translateY(0); }
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 2rem;
    animation: fadeUp 0.9s 0.1s cubic-bezier(.22,1,.36,1) both;
}

.badge-dot {
    width: 7px; height: 7px;
    background: #6366f1;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

.landing-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 7vw, 5.5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin-bottom: 1.5rem;
    animation: fadeUp 0.9s 0.2s cubic-bezier(.22,1,.36,1) both;
}

.landing-title .highlight {
    background: linear-gradient(135deg, #6366f1, #ec4899, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.landing-subtitle {
    font-size: 1.1rem;
    font-weight: 300;
    line-height: 1.7;
    color: rgba(232,232,240,0.65);
    max-width: 580px;
    margin: 0 auto 2.8rem;
    animation: fadeUp 0.9s 0.3s cubic-bezier(.22,1,.36,1) both;
}

/* ── Feature pills ── */
.features-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 12px;
    margin-bottom: 3rem;
    animation: fadeUp 0.9s 0.4s cubic-bezier(.22,1,.36,1) both;
}
.feature-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 0.88rem;
    font-weight: 400;
    color: rgba(232,232,240,0.8);
    backdrop-filter: blur(8px);
    transition: border-color 0.25s, background 0.25s;
}
.feature-pill:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.4);
}
.pill-icon { font-size: 1.1rem; }

/* ── Get Started Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 3rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s !important;
    box-shadow: 0 0 40px rgba(99,102,241,0.4), 0 8px 32px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 0 60px rgba(99,102,241,0.55), 0 12px 40px rgba(0,0,0,0.4) !important;
    opacity: 0.95 !important;
}
.stButton > button:active { transform: translateY(0) scale(0.99) !important; }

/* ── CHAT PAGE ── */
.chat-page { padding: 0 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif !important;
    color: #e8e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label { color: rgba(232,232,240,0.75) !important; }
[data-testid="stSidebar"] .stFileUploader label { color: rgba(232,232,240,0.7) !important; }

/* Top bar branding — fixed at top, never scrolls */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 2.5rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    background: rgba(9,9,15,0.8);
    backdrop-filter: blur(12px);
    position: sticky;
    top: 0;
    z-index: 100;
    flex-shrink: 0;
}

/* Chat layout: only the messages area scrolls; header and input stay fixed */
section.main .block-container {
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
    max-width: 100% !important;
}
/* Header section (summary, info, welcome) does not scroll */
section.main .block-container [data-testid="stVerticalBlock"]:not(:has([data-testid="stChatMessage"])) {
    flex-shrink: 0 !important;
}
/* Messages container is the only scrollable area */
section.main .block-container [data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"]) {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 0 1.5rem 1rem !important;
}
/* Chat input fixed at bottom */
section.main [data-testid="stChatInputContainer"] {
    flex-shrink: 0 !important;
}
.topbar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.8rem;
    color: rgba(232,232,240,0.5);
}
.status-dot {
    width: 7px; height: 7px;
    background: #34d399;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

/* Chat container */
.chat-area {
    max-width: 820px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 6rem;
}

/* Info banner */
[data-testid="stAlert"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    color: rgba(165,180,252,0.9) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.25rem 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.97rem !important;
    line-height: 1.7 !important;
    color: #e8e8f0 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.85rem 1.2rem !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 0.85rem 1.2rem !important;
}

/* Chat input — stays at bottom, does not scroll */
[data-testid="stChatInputContainer"] {
    background: rgba(15,15,26,0.95) !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    backdrop-filter: blur(12px) !important;
    padding: 1rem 1.5rem !important;
}
[data-testid="stChatInputContainer"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
[data-testid="stChatInputContainer"] button {
    background: linear-gradient(135deg, #6366f1, #ec4899) !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Expander (Sources) */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: rgba(165,180,252,0.8) !important; font-size: 0.85rem !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(99,102,241,0.4) !important; }

/* Radio buttons */
[data-testid="stRadio"] > div { gap: 8px !important; }
[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
    transition: all 0.2s !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 100px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.4); }
</style>
""", unsafe_allow_html=True)

# ─── Grain overlay (rendered once) ──────────────────────────────────────────
st.html('<div class="grain"></div>')

# ════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.started:
    st.html("""
    <div class="landing-wrap">
      <div class="landing-content">
        <div class="badge"><span class="badge-dot"></span>Intelligent Research Agent</div>

        <h1 class="landing-title">
          Meet <span class="highlight">NEO-AI</span><br/>
          Your AI Partner
        </h1>

        <p class="landing-subtitle">
          Neo-AI combines <strong>Retrieval-Augmented Generation</strong> with
          <strong>real-time web search</strong> to give you precise, cited answers —
          whether from your uploaded documents or the live web.
        </p>

        <div class="features-row">
          <div class="feature-pill">
            <span class="pill-icon">📄</span> RAG over your documents
          </div>
          <div class="feature-pill">
            <span class="pill-icon">🌐</span> Live web search
          </div>
          <div class="feature-pill">
            <span class="pill-icon">🧠</span> Conversational memory
          </div>
          <div class="feature-pill">
            <span class="pill-icon">⚡</span> Concise &amp; detailed modes
          </div>
          <div class="feature-pill">
            <span class="pill-icon">🔍</span> Source citations
          </div>
          <div class="feature-pill">
            <span class="pill-icon">📊</span> PDF · DOCX · CSV · TXT
          </div>
        </div>
      </div>
    </div>
    """)

    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✦  Get Started", use_container_width=True):
            st.session_state.started = True
            st.rerun()

    st.stop()


# ════════════════════════════════════════════════════════════════════════════
#  CHAT PAGE
# ════════════════════════════════════════════════════════════════════════════

# ── Top bar ──
st.html("""
<div class="topbar">
  <span class="topbar-brand">⚡ NeoAI</span>
  <span class="topbar-status"><span class="status-dot"></span>Agent online</span>
</div>
""")

# ── Sidebar ──

with st.sidebar:

    st.markdown("---")
    st.markdown("### ⚙️ Response mode")
    mode = st.radio(
        "mode",
        options=["concise", "detailed"],
        format_func=lambda x: "⚡ Concise" if x == "concise" else "📖 Detailed",
        key="mode",
        label_visibility="collapsed",
    )


    st.markdown("### 📂 Document")
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["pdf", "txt", "docx", "csv"],
        label_visibility="collapsed",
        help="Upload PDF, TXT, DOCX, or CSV — the agent will search it with RAG."
    )
    if uploaded_file is not None:
        file_key = (uploaded_file.name, uploaded_file.size)
        if st.session_state.last_processed_file != file_key:
            try:
                logger.info("File received: %s", uploaded_file.name)
                adapter = StreamlitFileAdapter(uploaded_file)
                vectorstore = process_document(adapter)
                if vectorstore is not None:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.last_processed_file = file_key
                    st.session_state.current_document_name = document_display_name(uploaded_file.name)
                    chunks = vectorstore.index.ntotal
                    st.success(f"✓ {uploaded_file.name} ({chunks} chunks)")
                else:
                    st.error("Failed to process. Try another file.")
            except ValueError as e:
                logger.warning("Document validation failed: %s", e)
                st.error(f"Invalid file: {e}")
            except DocumentProcessingError as e:
                logger.exception("Document processing failed")
                st.error(e.message if hasattr(e, "message") else str(e))
            except Exception:
                logger.exception("Unexpected error during document processing")
                st.error("Something went wrong. Please try again.")
        else:
            st.success(f"✓ {uploaded_file.name} loaded")

    # Summarize button when a document is indexed (does not modify vectorstore)
    if st.session_state.vectorstore and st.session_state.current_document_name and uploaded_file is not None:
        if st.button("📄 Summarize the document", key="sidebar_summarize", use_container_width=True):
            try:
                with st.spinner("Summarizing…"):
                    adapter = StreamlitFileAdapter(uploaded_file)
                    summary_text = summarize_document(adapter)
                    st.session_state.last_summary = {
                        "title": st.session_state.current_document_name,
                        "text": summary_text,
                    }
                st.rerun()
            except ValueError as e:
                st.error(f"Invalid file: {e}")
            except Exception as e:
                logger.exception("Summarize error")
                st.error("Summarization failed. Please try again.")

    st.markdown("---")
    st.markdown("### 📋 Summarize")
    st.caption("Summarize a document without adding it to the index. Upload PDF or TXT below.")
    summarize_file = st.file_uploader(
        "Document to summarize. Please Upload document with less than 5 pages",
        type=["pdf", "txt"],
        key="summarize_uploader",
        label_visibility="collapsed",
        help="Upload a PDF or TXT to summarize. This does not modify the RAG index.",
    )
    if summarize_file is not None:
        if st.button("Summarize this document", key="summarize_upload_btn", use_container_width=True):
            try:
                with st.spinner("Summarizing…"):
                    adapter = StreamlitFileAdapter(summarize_file)
                    summary_text = summarize_document(adapter)
                    st.session_state.last_summary = {
                        "title": document_display_name(summarize_file.name),
                        "text": summary_text,
                    }
                st.rerun()
            except ValueError as e:
                st.error(f"Invalid file: {e}")
            except Exception as e:
                logger.exception("Summarize error")
                st.error("Summarization failed. Please try again.")



    st.markdown("---")
    st.markdown("### 🤖 How it works")
    st.html("""
<div style="font-size:0.82rem; color:rgba(232,232,240,0.5); line-height:1.65;">

<strong>1. RAG</strong> — Searches your uploaded document using semantic vector retrieval.

<strong>2. Web Search</strong> — Falls back to live internet results when needed.

<strong>3. Memory</strong> — Maintains context throughout your conversation.

</div>
""")

    st.markdown("---")
    if st.button("← Back to Home", use_container_width=True):
        st.session_state.started = False
        st.rerun()

# ── Chat area: header (fixed) + messages (scrollable) ──
# Header: summary, info, welcome — does not scroll
with st.container():
    if st.session_state.last_summary:
        with st.expander(f"📋 Summary: {st.session_state.last_summary['title']}", expanded=True):
            st.markdown(st.session_state.last_summary["text"])
        if st.button("Clear summary", key="clear_summary"):
            st.session_state.last_summary = None
            st.rerun()

    if not st.session_state.vectorstore:
        st.info(
            "💡 No document loaded — Neo-AI will answer using web search and its knowledge. "
            "Upload a file in the sidebar to enable RAG over your own data."
        )

    # Welcome / "Ask me anything" — always visible at top, never scrolls
    st.html("""
<div class="chat-header-welcome" style="
    text-align:center;
    padding: 2rem 1rem 1.5rem;
    color: rgba(232,232,240,0.35);
    font-size: 0.92rem;
    line-height: 1.7;
">
  <div style="font-size:2.5rem; margin-bottom:0.75rem;">⚡</div>
  <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:rgba(232,232,240,0.6); margin-bottom:0.4rem;">
    Ask me anything
  </div>
  <div style="font-size:0.88rem;">I'll search your documents, browse the web, and synthesise a cited answer for you.</div>
</div>
""")

# Messages only — this block is the one that scrolls
with st.container():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔗 Sources"):
                    st.json(msg["sources"])

# ── Chat input ──
if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = answer_query(
                    user_query=prompt,
                    mode=mode,
                    thread_id=st.session_state.thread_id,
                    vectorstore=st.session_state.vectorstore,
                    document_uploaded=True if st.session_state.vectorstore is not None else False,
                    document_name=st.session_state.current_document_name,
                )
                answer = result.get("answer", "")
                sources = result.get("sources")
                st.markdown(answer)
                if sources:
                    with st.expander("🔗 Sources"):
                        st.json(sources)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                logger.exception("Chat error")
                st.error("Sorry, something went wrong. Please try again.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": str(e),
                    "sources": None,
                })