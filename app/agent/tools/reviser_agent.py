"""
Reviser Agent: rewrites a RAG answer that was judged well-grounded but weak on
tone (or with small factuality gaps), using the SAME retrieved context plus the
Evaluator's feedback. It never invents new medical facts - it only trims
unsupported claims, adjusts tone, and adds a consult-a-professional note where
appropriate. Invoked by the graph on the "revise" branch instead of a full
re-query.
"""

from langchain_core.prompts import ChatPromptTemplate

from app.config import properties_setup as settings
from app.llm.openai_llm import invoke_prompt
from app.prompts.reviser_prompt import REVISER_SYSTEM_PROMPT, REVISER_HUMAN_PROMPT
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", REVISER_SYSTEM_PROMPT),
    ("human", REVISER_HUMAN_PROMPT),
])


def revise_answer(question: str, answer: str, context: str, issues: str) -> str:
    """Returns the revised answer, or the original answer on any failure."""
    revised = invoke_prompt(
        PROMPT,
        {
            "question": question,
            "answer": answer,
            "context": context,
            "issues": issues or "(none specified - improve clarity and tone)",
        },
        temperature=settings.REVISER_TEMPERATURE,
        max_tokens=settings.REVISER_MAX_TOKENS,
        reasoning_max_tokens=settings.REVISER_REASONING_MAX_TOKENS,
    )
    if not revised:
        logger.warning("Reviser returned empty output, keeping original answer.")
        return answer
    logger.info("Reviser rewrote the answer (issues=%r)", issues)
    return revised
