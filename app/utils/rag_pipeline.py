"""
RAG pipeline: generate answers from retrieved document context.

Uses the runtime context's vectorstore for retrieval. Handles missing
vectorstore (no document uploaded) and falls back to web search when
the LLM returns no content.
"""
from langgraph.runtime import get_runtime

from app.components.models.llm import load_groq_client
from app.utils.web_search import web_search
from app.utils.logger.logger import get_logger
from app.exceptions import RAGError

logger = get_logger()

MODE_CONCISE = "concise"
MODE_DETAILED = "detailed"


def generate_answer(question: str, mode: str) -> str:
    """
    Generate an answer using RAG (retrieved context) and optional web fallback.

    Args:
        question: User question.
        mode: "concise" or "detailed" to control response length.

    Returns:
        Answer string. If no document is loaded, returns a short message
        suggesting upload or use of general search.
    """
    try:
        runtime_res = get_runtime()
        vectorstore = runtime_res.context.get("vectorstore")

        if vectorstore is None:
            logger.info("RAG called but no document loaded; suggesting user upload or use web search.")
            return (
                "No document has been uploaded. Please upload a document in the sidebar to query it, "
                "or ask a general question and I can search the web."
            )

        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)

        context = "\n".join([doc.page_content for doc in docs])

        logger.info(f"Context retrieved are : {context}")

        llm = load_groq_client()

        if mode == MODE_CONCISE:
            prompt = f"""Answer briefly using this context.

Context:
{context}

Question:
{question}
"""
        else:
            prompt = f"""Give a detailed explanation.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        if not response.content:
            web_results = web_search(question)
            prompt = f"""Answer using web data.

{web_results}

Question:
{question}
"""
            response = llm.invoke(prompt)

        return response.content or "I couldn't generate a response. Please try rephrasing or uploading a document."

    except RAGError:
        raise
    except Exception as e:
        logger.exception("RAG error: %s", e)
        raise RAGError("Error generating response from documents.", details=str(e)) from e