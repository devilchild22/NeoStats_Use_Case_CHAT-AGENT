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

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        context = "\n".join([doc.page_content for doc in docs])

        logger.info(f"Context retrieved are : {context}")

        llm = load_groq_client()

        if mode == MODE_CONCISE:
            prompt = f"""
You are an AI assistant answering questions based ONLY on the provided document context.

INSTRUCTIONS:
- Use only the information from the context below.
- Do NOT add external knowledge.
- If the answer is not present in the context, say: "The uploaded document does not contain information about this."

RESPONSE STYLE:
- Keep the answer very short.
- Maximum length: 3 lines.
- Use simple and direct language.
- Avoid long explanations.
- If bullet points are needed, keep them extremely brief.

CONTEXT:
{context}

QUESTION:
{question}

"""
        else:
            prompt = f"""
You are an AI assistant answering questions using the provided document context.

INSTRUCTIONS:
- Use only the information from the context below.
- Do NOT add external knowledge.
- If the context does not contain the answer, say: "The uploaded document does not contain information about this."

RESPONSE STYLE:
- Provide a clear and well-structured explanation.
- Use proper formatting suitable for UI display.
- Use headings, bullet points, or numbered steps where helpful.
- Keep the explanation easy to understand.

CONTEXT:
{context}

QUESTION:
{question}

"""
        response = llm.invoke(prompt)

        logger.info(f"Response from LLM for RAG pipeline: {response.content}")

        return response.content

    except RAGError:
        raise
    except Exception as e:
        logger.exception("RAG error: %s", e)
        raise RAGError("Error generating response from documents.", details=str(e)) from e