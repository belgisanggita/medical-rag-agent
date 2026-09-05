"""
Evaluator Agent: LLM-as-judge. Checks the RAG answer for factual grounding
against the retrieved context and returns a confidence score in [0, 1].
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from app.config import properties_setup as settings
from app.llm.openai_llm import invoke_prompt
from app.prompts.evaluator_prompt import EVALUATOR_SYSTEM_PROMPT, EVALUATOR_HUMAN_PROMPT
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", EVALUATOR_SYSTEM_PROMPT),
    ("human", EVALUATOR_HUMAN_PROMPT),
])

DEFAULT_SCORE_ON_PARSE_FAILURE = 0.5


def evaluate_answer(question: str, answer: str, context: str) -> float:
    """Returns a confidence score between 0 and 1."""
    if not context:
        return 0.0

    raw = invoke_prompt(
        PROMPT,
        {"context": context, "question": question, "answer": answer},
        temperature=settings.EVALUATOR_TEMPERATURE,
        max_tokens=settings.EVALUATOR_MAX_TOKENS,
        reasoning_max_tokens=settings.EVALUATOR_REASONING_MAX_TOKENS,
    )

    match = re.search(r"(\d*\.?\d+)", raw)
    if not match:
        logger.warning(f"Evaluator returned unparseable score: {raw!r}, defaulting to {DEFAULT_SCORE_ON_PARSE_FAILURE}")
        return DEFAULT_SCORE_ON_PARSE_FAILURE

    return max(0.0, min(1.0, float(match.group(1))))
