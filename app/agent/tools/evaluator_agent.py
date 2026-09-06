"""
Evaluator Agent: LLM-as-judge. Scores the RAG answer on two axes against the
retrieved context:

  * factuality - is the answer grounded in the context (no hallucination)?
  * tone       - is it appropriate for a medical assistant?

Returns a dict {"factuality": float, "tone": float, "issues": str}. The graph
uses these scores (via planner_agent.decide_after_evaluation) to choose
between accept / revise / re-query / escalate.
"""

import json
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

# Neutral fallback when the judge output can't be parsed - mid-scale so the
# graph neither blindly trusts nor blindly rejects the answer.
DEFAULT_ON_PARSE_FAILURE = {"factuality": 0.5, "tone": 0.5, "issues": "evaluator output unparseable"}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def evaluate_answer(question: str, answer: str, context: str) -> dict:
    """Returns {"factuality": [0,1], "tone": [0,1], "issues": str}."""
    if not context:
        return {"factuality": 0.0, "tone": 0.5, "issues": "no context was retrieved"}

    raw = invoke_prompt(
        PROMPT,
        {"context": context, "question": question, "answer": answer},
        temperature=settings.EVALUATOR_TEMPERATURE,
        max_tokens=settings.EVALUATOR_MAX_TOKENS,
        reasoning_max_tokens=settings.EVALUATOR_REASONING_MAX_TOKENS,
    )

    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        logger.warning(f"Evaluator returned no JSON object: {raw!r}, using default scores")
        return dict(DEFAULT_ON_PARSE_FAILURE)

    try:
        data = json.loads(match.group(0))
        result = {
            "factuality": _clamp(data.get("factuality", 0.5)),
            "tone": _clamp(data.get("tone", 0.5)),
            "issues": str(data.get("issues", "")).strip(),
        }
        logger.info(
            "Evaluator: factuality=%.2f tone=%.2f issues=%r",
            result["factuality"], result["tone"], result["issues"],
        )
        return result
    except (ValueError, TypeError) as e:
        logger.warning(f"Evaluator JSON parse failed ({e}): {raw!r}, using default scores")
        return dict(DEFAULT_ON_PARSE_FAILURE)
