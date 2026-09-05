"""
Summarizer Agent: condenses the conversation so far into a short running
summary, so older turns don't need to be re-sent in full on every request
(keeps prompts small - see RAG_RECENT_TURNS in config/properties.env).
"""

from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import properties_setup as settings
from app.llm.openai_llm import invoke_prompt
from app.prompts.summarizer_prompt import (
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_PREVIOUS_SUMMARY_PROMPT,
    SUMMARIZER_HUMAN_PROMPT,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUMMARIZER_SYSTEM_PROMPT),
    ("human", SUMMARIZER_PREVIOUS_SUMMARY_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", SUMMARIZER_HUMAN_PROMPT),
])


def summarize_conversation(chat_history: List[BaseMessage], previous_summary: str) -> str:
    """Returns an updated running summary string."""
    if not chat_history:
        return previous_summary

    return invoke_prompt(
        PROMPT,
        {
            "previous_summary": previous_summary or "(none yet)",
            "chat_history": chat_history,
        },
        temperature=settings.SUMMARIZER_TEMPERATURE,
        max_tokens=settings.SUMMARIZER_MAX_TOKENS,
        reasoning_max_tokens=settings.SUMMARIZER_REASONING_MAX_TOKENS,
    )
