"""
LangGraph orchestration for the medical multi-agent assistant.

Nodes: planner, rag, evaluator, reviser, escalate, meta, summarizer.

Flow:

    planner ─┬─ small_talk ─────────────────────────────► END   (fixed redirect)
             ├─ meta ──────────► meta ──────────────────► END   (summary only)
             └─ medical ──────► rag ──► evaluator ─┬─ accept ──► summarizer ► END
                                  ▲                ├─ revise ──► reviser ───► summarizer ► END
                                  └── retry ───────┤
                                                   └─ escalate ► escalate ──► summarizer ► END

The planner (app/agent/tools/planner_agent.py) owns BOTH routing decisions:
which agents to activate for a message, and - after evaluation - whether to
re-query, revise, accept, or escalate based on the confidence/tone scores.
"""

from typing import TypedDict, List

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from app.agent.tools.planner_agent import decide_next_action, decide_after_evaluation
from app.agent.tools.rag_agent import generate_answer
from app.agent.tools.evaluator_agent import evaluate_answer
from app.agent.tools.reviser_agent import revise_answer
from app.agent.tools.summarizer_agent import summarize_conversation


class AgentState(TypedDict, total=False):
    question: str
    chat_history: List[BaseMessage]  # turns prior to `question`, oldest first
    summary: str

    context: str
    answer: str

    # evaluator output
    factuality: float
    tone: float
    eval_issues: str
    confidence: float  # kept as an alias of `factuality` for the UI / back-compat

    # control
    intent: str            # "medical" | "meta" | "small_talk"
    plan: List[str]        # agents activated this turn (shown in the UI)
    rag_attempts: int      # how many times RAG has generated an answer this turn
    revised: bool          # the reviser has already run this turn
    escalated: bool        # answer flagged as low-confidence to the user


OFF_TOPIC_ANSWER = (
    "I can only answer questions related to the medical topics covered in "
    "the Gale Encyclopedia of Medicine. Please ask me something about a "
    "medical condition, treatment, procedure, or term.\n\n"
    "Saya hanya bisa menjawab pertanyaan seputar topik medis yang ada di "
    "Gale Encyclopedia of Medicine. Silakan tanyakan sesuatu tentang "
    "kondisi, pengobatan, prosedur, atau istilah medis."
)

ESCALATION_BANNER = (
    "⚠️ **Jawaban di bawah ini tidak dapat diverifikasi dengan tingkat "
    "keyakinan yang memadai terhadap sumber (Gale Encyclopedia of Medicine).** "
    "Mohon konfirmasikan ke tenaga medis profesional sebelum mengambil "
    "tindakan apa pun.\n\n"
    "⚠️ *The answer below could not be verified with sufficient confidence "
    "against the source. Please confirm with a qualified health professional "
    "before acting on it.*\n\n---\n\n"
)

NO_HISTORY_TO_SUMMARIZE = (
    "Belum ada percakapan sebelumnya untuk diringkas. / "
    "There is no earlier conversation to summarize yet."
)


def planner_node(state: AgentState) -> AgentState:
    resolved_question, intent, plan = decide_next_action(state)
    patch: AgentState = {"question": resolved_question, "intent": intent, "plan": plan}
    if intent == "small_talk":
        patch["answer"] = OFF_TOPIC_ANSWER
    return {**state, **patch}


def meta_node(state: AgentState) -> AgentState:
    # User asked about the conversation itself - answer with a fresh summary.
    new_summary = summarize_conversation(
        chat_history=state.get("chat_history", []),
        previous_summary=state.get("summary", ""),
    )
    answer = new_summary.strip() if new_summary and new_summary.strip() else NO_HISTORY_TO_SUMMARIZE
    return {**state, "summary": new_summary or state.get("summary", ""), "answer": answer}


def rag_node(state: AgentState) -> AgentState:
    answer, context = generate_answer(
        question=state["question"],
        summary=state.get("summary", ""),
        chat_history=state.get("chat_history", []),
    )
    return {
        **state,
        "answer": answer,
        "context": context,
        "rag_attempts": state.get("rag_attempts", 0) + 1,
    }


def evaluator_node(state: AgentState) -> AgentState:
    ev = evaluate_answer(
        question=state["question"],
        answer=state["answer"],
        context=state.get("context", ""),
    )
    return {
        **state,
        "factuality": ev["factuality"],
        "tone": ev["tone"],
        "eval_issues": ev["issues"],
        "confidence": ev["factuality"],
    }


def _plan_with(plan: List[str], agent: str) -> List[str]:
    """Insert `agent` just before the trailing 'summarizer' (or append)."""
    if agent in plan:
        return plan
    if plan and plan[-1] == "summarizer":
        return plan[:-1] + [agent, "summarizer"]
    return plan + [agent]


def reviser_node(state: AgentState) -> AgentState:
    new_answer = revise_answer(
        question=state["question"],
        answer=state["answer"],
        context=state.get("context", ""),
        issues=state.get("eval_issues", ""),
    )
    return {
        **state,
        "answer": new_answer,
        "revised": True,
        "plan": _plan_with(state.get("plan", []), "reviser"),
    }


def escalate_node(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": ESCALATION_BANNER + state.get("answer", ""),
        "escalated": True,
        "plan": _plan_with(state.get("plan", []), "escalate"),
    }


def summarizer_node(state: AgentState) -> AgentState:
    new_summary = summarize_conversation(
        chat_history=state.get("chat_history", []),
        previous_summary=state.get("summary", ""),
    )
    return {**state, "summary": new_summary}


def _route_after_planner(state: AgentState) -> str:
    intent = state.get("intent", "medical")
    if intent == "small_talk":
        return "off_topic"
    if intent == "meta":
        return "meta"
    return "rag"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("meta", meta_node)
    graph.add_node("rag", rag_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("reviser", reviser_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"rag": "rag", "meta": "meta", "off_topic": END},
    )

    graph.add_edge("meta", END)
    graph.add_edge("rag", "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        decide_after_evaluation,
        {
            "accept": "summarizer",
            "revise": "reviser",
            "retry": "rag",
            "escalate": "escalate",
        },
    )

    graph.add_edge("reviser", "summarizer")
    graph.add_edge("escalate", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
