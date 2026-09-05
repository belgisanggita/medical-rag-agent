"""
RAG Agent: retrieve relevant chunks from Qdrant and generate an answer.
"""

from typing import List, Tuple

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import properties_setup as settings
from app.index.qdrant_index import embed_query, search
from app.llm.openai_llm import invoke_prompt
from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT, RAG_SUMMARY_PROMPT, RAG_HUMAN_PROMPT
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("system", RAG_SUMMARY_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", RAG_HUMAN_PROMPT),
])

NO_CONTEXT_ANSWER = (
    "Sorry, I couldn't find relevant information in the source to answer this question. / "
    "Maaf, saya tidak menemukan informasi yang relevan di sumber untuk menjawab pertanyaan ini."
)
EMPTY_LLM_RESPONSE_ANSWER = (
    "Sorry, something went wrong while generating an answer. Please try asking again. / "
    "Maaf, terjadi kesalahan saat membuat jawaban. Silakan coba tanyakan lagi."
)


def retrieve_context(question: str, top_k: int = None) -> str:
    """Fetch the most relevant chunks from Qdrant for this question."""
    vector = embed_query(question)
    results = search(vector, top_k=top_k or settings.RAG_TOP_K)
    if not results:
        return ""

    parts = []
    for r in results:
        label = r.get("entry", "")
        if r.get("type") == "content":
            label = f"{label} - {r.get('section', '')}"
        elif r.get("type") == "glossary":
            label = f"{label} - {r.get('term', '')}"
        parts.append(f"[{label}]\n{r['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, summary: str, chat_history: List[BaseMessage]) -> Tuple[str, str]:
    """
    Returns (answer, context_used) so the Evaluator can check the answer
    against the same context that was retrieved. chat_history holds turns
    prior to this question (the caller passes the current question
    separately, not appended to the history).
    """
    context = retrieve_context(question)
    if not context:
        return NO_CONTEXT_ANSWER, ""

    answer = invoke_prompt(
        PROMPT,
        {
            "summary": summary or "(none yet)",
            "chat_history": chat_history[-settings.RAG_RECENT_TURNS:],
            "context": context,
            "question": question,
        },
        temperature=settings.RAG_TEMPERATURE,
        max_tokens=settings.RAG_MAX_TOKENS,
        reasoning_max_tokens=settings.RAG_REASONING_MAX_TOKENS,
    )
    return answer or EMPTY_LLM_RESPONSE_ANSWER, context
