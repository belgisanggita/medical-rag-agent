"""
LangGraph orchestration: 4 nodes (planner, rag, evaluator, summarizer).

Planner -> RAG -> Evaluator -> Summarizer, with a conditional loop back to
RAG when the Evaluator's confidence score is too low (re-query), and a
conditional path straight to END when the Planner decides the question has
nothing to do with the medical book (no RAG call, no fabricated answer -
just a fixed prompt to ask something on-topic instead).

Each *_tools module contains the actual work for its agent; this file only
wires them together as a graph.
"""

from typing import TypedDict, List

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from app.agent.tools.planner_agent import decide_next_action
from app.agent.tools.rag_agent import generate_answer
from app.agent.tools.evaluator_agent import evaluate_answer
from app.agent.tools.summarizer_agent import summarize_conversation
from app.config import properties_setup as settings


class AgentState(TypedDict, total=False):
    question: str
    chat_history: List[BaseMessage]  # turns prior to `question`, oldest first
    summary: str

    context: str
    answer: str
    confidence: float
    retry_count: int
    next_action: str  # set by planner: "rag" | "off_topic"


CONFIDENCE_THRESHOLD = settings.CONFIDENCE_THRESHOLD
MAX_RETRIES = settings.MAX_RETRIES

# Off-topic must never get a fabricated answer - just a fixed, bilingual
# redirect back to what this assistant can actually help with.
OFF_TOPIC_ANSWER = (
    "I can only answer questions related to the medical topics covered in "
    "the Gale Encyclopedia of Medicine. Please ask me something about a "
    "medical condition, treatment, procedure, or term.\n\n"
    "Saya hanya bisa menjawab pertanyaan seputar topik medis yang ada di "
    "Gale Encyclopedia of Medicine. Silakan tanyakan sesuatu tentang "
    "kondisi, pengobatan, prosedur, atau istilah medis."
)


def planner_node(state: AgentState) -> AgentState:
    next_action = decide_next_action(state)
    if next_action == "off_topic":
        return {**state, "next_action": next_action, "answer": OFF_TOPIC_ANSWER}
    return {**state, "next_action": next_action}


def rag_node(state: AgentState) -> AgentState:
    answer, context = generate_answer(
        question=state["question"],
        summary=state.get("summary", ""),
        chat_history=state.get("chat_history", []),
    )
    return {**state, "answer": answer, "context": context}


def evaluator_node(state: AgentState) -> AgentState:
    confidence = evaluate_answer(
        question=state["question"],
        answer=state["answer"],
        context=state.get("context", ""),
    )
    # increment here so the retry counter actually advances each loop,
    # otherwise the retry -> rag -> evaluator cycle would never terminate
    retries = state.get("retry_count", 0) + 1
    return {**state, "confidence": confidence, "retry_count": retries}


def summarizer_node(state: AgentState) -> AgentState:
    new_summary = summarize_conversation(
        chat_history=state.get("chat_history", []),
        previous_summary=state.get("summary", ""),
    )
    return {**state, "summary": new_summary}


def _route_after_planner(state: AgentState) -> str:
    return state.get("next_action", "rag")


def _route_after_evaluator(state: AgentState) -> str:
    low_confidence = state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
    can_retry = state.get("retry_count", 0) < MAX_RETRIES
    if low_confidence and can_retry:
        return "retry"
    return "done"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("rag", rag_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"rag": "rag", "off_topic": END},
    )

    graph.add_edge("rag", "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {"retry": "rag", "done": "summarizer"},
    )

    graph.add_edge("summarizer", END)

    return graph.compile()
