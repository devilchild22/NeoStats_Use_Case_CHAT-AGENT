"""
Financial AI agent with RAG and web search tools.

Exposes answer_query() for the Streamlit app. Uses LangGraph checkpointing
for per-thread conversation state.
"""
from typing import Any, Dict

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.components.models.llm import load_groq_client
from app.components.prompt.prompt import AGENT_PROMPT
from app.utils.logger.logger import get_logger
from app.utils.rag_pipeline import generate_answer
from app.utils.web_search import web_search
from app.exceptions import AgentError

logger = get_logger()
client = load_groq_client()
memory = InMemorySaver()


@tool(
    description=(
        "Use this tool to answer questions about financial documents uploaded by the user. "
        "It retrieves relevant information from the document knowledge base. "
        "Do NOT use this tool for general internet questions."
    ),
)
def rag_tool(question: str, mode: str) -> str:
    """Retrieve answers from uploaded financial documents using RAG."""
    try:
        return generate_answer(question, mode)
    except Exception as e:
        logger.warning("RAG tool error: %s", e)
        return f"Error retrieving information from uploaded documents: {str(e)}"


@tool(
    description=(
        "Use this tool to search the internet for financial news, market updates, "
        "or information that is NOT available in the uploaded documents."
    ),
)
def internet_search(query: str) -> str:
    """Search the internet for financial information."""
    try:
        return web_search(query)
    except Exception as e:
        logger.error("Error performing internet search: %s", e)
        raise


def initialize_agent():
    """
    Create and return the financial AI agent with RAG and web search tools.

    Returns:
        The configured LangGraph agent.

    Raises:
        AgentError: If agent creation fails.
    """
    try:
        tools = [rag_tool, internet_search]
        agent = create_agent(
            name="RAG and WEB SEARCH ASSISTANT",
            model=client,
            tools=tools,
            system_prompt=AGENT_PROMPT,
            checkpointer=memory,
        )
        return agent
    except Exception as e:
        logger.error("Error creating financial agent: %s", e)
        raise AgentError("Failed to create agent.", details=str(e)) from e


agent = initialize_agent()


def _extract_ai_message(messages) -> str:
    """Return the content of the last AI message, or a fallback string."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "Sorry, I could not respond at this time. Please try again later."


def answer_query(
    user_query: str,
    mode: str,
    thread_id: str,
    vectorstore: Any,
    document_uploaded: bool,
) -> Dict[str, Any]:
    """
    Run the agent on a user query and return the answer and optional sources.

    Args:
        user_query: The user's question.
        mode: "concise" or "detailed".
        thread_id: Id for conversation state (e.g. user name).
        vectorstore: FAISS vectorstore or None if no document uploaded.
        document_uploaded: Whether a document is currently loaded.

    Returns:
        Dict with "answer" (str) and "sources" (optional list).

    Raises:
        AgentError: If invocation or extraction fails.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        context = {"vectorstore": vectorstore}
        inputs = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"The user query is: {user_query}\n"
                        f"The mode is: {mode}\n"
                        f"Document uploaded: {document_uploaded}"
                    ),
                }
            ]
        }

        result = agent.invoke(inputs, config=config, context=context)
        final_response = _extract_ai_message(result["messages"])
        return {"answer": final_response, "sources": None}
    except AgentError:
        raise
    except Exception as e:
        logger.exception("Error in answer_query")
        raise AgentError("Failed to get answer from agent.", details=str(e)) from e

