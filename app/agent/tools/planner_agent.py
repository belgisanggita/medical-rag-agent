"""
Planner Agent: decides whether the user's message is related to the medical
book (-> "rag", must be looked up and answered) or not (-> "off_topic", must
never get a fabricated answer - graph.py routes this straight to a fixed
redirect message instead of calling RAG).

The other interesting decision (retry vs done after evaluation) lives in
the graph's conditional edges in app/agent/graph.py - this is only the
upfront routing before RAG runs.
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

from app.config import properties_setup as settings
from app.llm.openai_llm import invoke_prompt
from app.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_HUMAN_PROMPT
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

Action = Literal["rag", "off_topic"]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    ("human", PLANNER_HUMAN_PROMPT),
])

VALID_ACTIONS = {"rag", "off_topic"}
DEFAULT_ACTION: Action = "rag"


def decide_next_action(state: dict) -> Action:
    question = state.get("question", "")
    if not question.strip():
        return "off_topic"

    raw = invoke_prompt(
        PROMPT,
        {"question": question},
        temperature=settings.PLANNER_TEMPERATURE,
        max_tokens=settings.PLANNER_MAX_TOKENS,
        reasoning_max_tokens=settings.PLANNER_REASONING_MAX_TOKENS,
    )

    action = raw.strip().lower()
    if action not in VALID_ACTIONS:
        logger.warning(f"Planner returned unrecognized action: {action!r}, defaulting to '{DEFAULT_ACTION}'")
        return DEFAULT_ACTION
    return action
