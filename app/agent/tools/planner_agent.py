"""
Planner Agent. Owns two routing decisions in the graph:

1. decide_next_action(state) - runs one LLM call that (a) rewrites the user's
   message into a standalone question by resolving references to earlier turns,
   and (b) classifies its intent as:
       "medical"    -> full RAG pipeline (rag -> evaluator -> ...)
       "meta"       -> summarizer only (user asked about the conversation)
       "small_talk" -> fixed off-topic redirect, no RAG, no fabricated answer
   It also returns the ordered list of agents it plans to activate, which the
   UI displays.

2. decide_after_evaluation(state) - rule-based decision taken after the
   Evaluator has scored the answer:
       "accept"   -> answer is grounded and well-toned, finish
       "revise"   -> grounded but tone/minor issues, send to Reviser
       "retry"    -> not grounded, budget left, re-query RAG
       "escalate" -> not grounded, out of retries, flag uncertainty to the user
"""

import re
from typing import List, Literal, Tuple

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import properties_setup as settings
from app.llm.openai_llm import invoke_prompt
from app.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_HUMAN_PROMPT
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

Intent = Literal["medical", "meta", "small_talk"]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", PLANNER_HUMAN_PROMPT),
])

VALID_INTENTS = {"medical", "meta", "small_talk"}
DEFAULT_INTENT: Intent = "medical"

# Which agents each intent activates (shown in the UI, extended at runtime by
# the reviser/escalate nodes when those branches fire).
PLAN_BY_INTENT = {
    "medical": ["planner", "rag", "evaluator", "summarizer"],
    "meta": ["planner", "summarizer"],
    "small_talk": ["planner"],
}


def decide_next_action(state: dict) -> Tuple[str, Intent, List[str]]:
    """Returns (resolved_question, intent, planned_agents)."""
    question = state.get("question", "")
    if not question.strip():
        return question, "small_talk", list(PLAN_BY_INTENT["small_talk"])

    chat_history: List[BaseMessage] = state.get("chat_history", [])

    raw = invoke_prompt(
        PROMPT,
        {"chat_history": chat_history[-settings.RAG_RECENT_TURNS:], "question": question},
        temperature=settings.PLANNER_TEMPERATURE,
        max_tokens=settings.PLANNER_MAX_TOKENS,
        reasoning_max_tokens=settings.PLANNER_REASONING_MAX_TOKENS,
    )

    question_match = re.search(r"QUESTION:\s*(.+)", raw or "")
    intent_match = re.search(r"INTENT:\s*(\w+)", raw or "")

    resolved_question = question_match.group(1).strip() if question_match else question
    intent = intent_match.group(1).strip().lower() if intent_match else DEFAULT_INTENT

    if intent not in VALID_INTENTS:
        logger.warning(f"Planner returned unrecognized intent: {intent!r}, defaulting to {DEFAULT_INTENT!r}")
        intent = DEFAULT_INTENT

    if resolved_question != question:
        logger.info(f"Planner rewrote question: {question!r} -> {resolved_question!r}")
    logger.info(f"Planner intent={intent!r}, plan={PLAN_BY_INTENT[intent]}")

    return resolved_question, intent, list(PLAN_BY_INTENT[intent])


def decide_after_evaluation(state: dict) -> Literal["accept", "revise", "retry", "escalate"]:
    """Rule-based post-evaluation routing based on the Evaluator's scores."""
    factuality = state.get("factuality", 1.0)
    tone = state.get("tone", 1.0)
    rag_attempts = state.get("rag_attempts", 1)

    fact_ok = factuality >= settings.CONFIDENCE_THRESHOLD
    tone_ok = tone >= settings.TONE_THRESHOLD

    if fact_ok and tone_ok:
        decision = "accept"
    elif fact_ok and not state.get("revised", False):
        # grounded but tone / minor issues - a targeted rewrite is enough
        decision = "revise"
    elif fact_ok:
        # already revised once; the answer is grounded, so ship it rather than
        # escalate a factually-correct answer over a residual tone nitpick
        decision = "accept"
    elif rag_attempts <= settings.MAX_RETRIES:
        decision = "retry"
    else:
        decision = "escalate"

    logger.info(
        "Planner post-eval: factuality=%.2f tone=%.2f rag_attempts=%d -> %s",
        factuality, tone, rag_attempts, decision,
    )
    return decision
