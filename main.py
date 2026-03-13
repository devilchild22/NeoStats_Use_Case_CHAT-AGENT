"""
Streamlit entry point for Neostas.

Provides a chat interface with optional document upload, concise/detailed mode,
and per-user conversation state. Run with:

    streamlit run main.py
"""
import streamlit as st
from app.components.agent.agent import answer_query
from app.utils.process_doc.processing import process_document
from app.utils.logger.logger import get_logger
from app.exceptions import DocumentProcessingError

logger = get_logger()

# Session state keys and defaults (single source of truth)
SESSION_KEYS = ("vectorstore", "messages", "thread_id", "last_processed_file")
SESSION_DEFAULTS = {
    "vectorstore": None,
    "messages": [],
    "thread_id": "streamlit-default",
    "last_processed_file": None,
}


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


st.set_page_config(page_title="Neostas", layout="wide")
init_session_state()

# Sidebar: file upload and mode
with st.sidebar:
    st.header("Document & settings")

    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, DOCX, CSV)",
        type=["pdf", "txt", "docx", "csv"],
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
                    chunks = vectorstore.index.ntotal
                    st.success(f"Document processed successfully ({chunks} chunks).")
                else:
                    st.error("Failed to process document. Please try another file.")
            except ValueError as e:
                logger.warning("Document validation failed: %s", e)
                st.error(f"Unsupported or invalid file: {e}")
            except DocumentProcessingError as e:
                logger.exception("Document processing failed")
                st.error(e.message if hasattr(e, "message") else str(e))
            except Exception as e:
                logger.exception("Unexpected error during document processing")
                st.error("Something went wrong processing your file. Please try again.")
        else:
            st.success("Document loaded. Ask a question in the chat.")

    st.divider()
    mode = st.radio(
        "Response mode",
        options=["concise", "detailed"],
        format_func=lambda x: "Concise" if x == "concise" else "Detailed",
        key="mode",
    )

# Main area: chat
st.title("CHAT-AGENT")

if not st.session_state.vectorstore:
    st.info("Upload a document (optional) in the sidebar to query your files, or ask general questions below.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.json(msg["sources"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_query(
                    user_query=prompt,
                    mode=mode,
                    thread_id=st.session_state.thread_id,
                    vectorstore=st.session_state.vectorstore,
                    document_uploaded=True if st.session_state.vectorstore is not None else False,
                )
                answer = result.get("answer", "")
                sources = result.get("sources")
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
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
