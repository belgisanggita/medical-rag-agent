"""
Planner Agent: in one LLM call, (1) rewrites the user's message into a
standalone question by resolving references to earlier turns (e.g. "apa
gejalanya?" right after discussing diabetes becomes "apa gejala diabetes?"),
so RAG's retrieval embeds the real topic instead of a bare pronoun/ellipsis
that would pull back unrelated chunks - and (2) decides whether that
question is related to the medical book ("rag") or not ("off_topic", which
graph.py routes straight to a fixed redirect message instead of calling
RAG - never a fabricated answer).

The other interesting decision (retry vs done after evaluation) lives in
the graph's conditional edges in app/agent/graph.py.
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

Action = Literal["rag", "off_topic"]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", PLANNER_HUMAN_PROMPT),
])

VALID_ACTIONS = {"rag", "off_topic"}
DEFAULT_ACTION: Action = "rag"


def decide_next_action(state: dict) -> Tuple[str, Action]:
    """Returns (resolved_question, action)."""
    question = state.get("question", "")
    if not question.strip():
        return question, "off_topic"

    chat_history: List[BaseMessage] = state.get("chat_history", [])

    raw = invoke_prompt(
        PROMPT,
        {"chat_history": chat_history[-settings.RAG_RECENT_TURNS:], "question": question},
        temperature=settings.PLANNER_TEMPERATURE,
        max_tokens=settings.PLANNER_MAX_TOKENS,
        reasoning_max_tokens=settings.PLANNER_REASONING_MAX_TOKENS,
    )

    question_match = re.search(r"QUESTION:\s*(.+)", raw)
    action_match = re.search(r"ACTION:\s*(\w+)", raw)

    resolved_question = question_match.group(1).strip() if question_match else question
    action = action_match.group(1).strip().lower() if action_match else DEFAULT_ACTION

    if action not in VALID_ACTIONS:
        logger.warning(f"Planner returned unrecognized action: {action!r}, defaulting to '{DEFAULT_ACTION}'")
        action = DEFAULT_ACTION

    if resolved_question != question:
        logger.info(f"Planner rewrote question: {question!r} -> {resolved_question!r}")

    return resolved_question, action
