"""
Streamlit entry point for the Medical RAG multi-agent assistant.

Run with:
    streamlit run app.py

This file owns the UI only. All orchestration logic (planner -> rag ->
evaluator -> summarizer) lives in app/agent/graph.py and is invoked here
as a plain Python function call - no HTTP layer involved.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from app.config import properties_setup as settings
from app.utils.logger import setup_logger
from app.agent.graph import build_graph
from app.index.qdrant_index import is_model_cached, warm_up_model
from ingest import ensure_ingested

logger = setup_logger(__name__)

st.set_page_config(page_title=settings.APP_NAME, page_icon="🩺")


@st.cache_resource(show_spinner=False)
def _warm_up_embedding_model() -> bool:
    """Pull the embedding model up front and say so, instead of letting the
    first question hang for minutes on a silent ~1 GB download.

    Returns True if this process had to download it. Like the ingest check,
    st.cache_resource keeps this to one call per server process.
    """
    downloading = not is_model_cached()
    message = (
        "⏬ Embedding Model Downloading... (This is only for first time)"
        if downloading
        else "Memuat embedding model..."
    )
    logger.info(
        "Warming up embedding model '%s' (cached=%s)...",
        settings.EMBEDDING_MODEL,
        not downloading,
    )
    with st.spinner(message):
        warm_up_model()
    return downloading


try:
    _first_download = _warm_up_embedding_model()
except Exception:
    logger.exception("Embedding model warm-up failed on startup.")
    st.error(
        f"Gagal menyiapkan embedding model '{settings.EMBEDDING_MODEL}'. "
        "Cek koneksi internet container dan logs/medical_generative.log."
    )
    st.stop()

# The cached call returns the same flag on every rerun, so gate the notice on
# session state - otherwise it would pop up again on each interaction.
if _first_download and not st.session_state.get("model_download_notified"):
    st.session_state.model_download_notified = True
    st.toast("Embedding model selesai diunduh dan siap dipakai.", icon="✅")


@st.cache_resource(show_spinner="Memeriksa index dokumen di Qdrant...")
def _ensure_ingested():
    # st.cache_resource keeps this to one call per server process (not per
    # session, not per rerun) - ensure_ingested() itself is also idempotent
    # via doc_id, so a restarted process still skips re-embedding instantly.
    if not settings.MEDICAL_PDF_PATH:
        logger.warning("MEDICAL_PDF_PATH is not set in config/properties.env - skipping auto-ingest.")
        return
    ensure_ingested(settings.MEDICAL_PDF_PATH)


try:
    _ensure_ingested()
except Exception:
    logger.exception("Auto-ingest failed on startup.")
    st.error("Gagal menyiapkan index dokumen. Cek logs/medical_generative.log untuk detail error.")
    st.stop()

# --- session state (replaces "history in Redis" - see conversation notes) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List[BaseMessage] (HumanMessage/AIMessage)

if "summary" not in st.session_state:
    st.session_state.summary = ""  # produced by the Summarizer agent

if "graph" not in st.session_state:
    # Build the LangGraph app once per session, reuse across turns.
    st.session_state.graph = build_graph()

if st.sidebar.button("🗑️ Reset chat"):
    logger.debug(
        "reset chat | clearing chat_history=%d msg(s) | summary=%r",
        len(st.session_state.chat_history),
        st.session_state.summary,
    )
    st.session_state.chat_history = []
    st.session_state.summary = ""
    st.rerun()


logger.debug(
    "session_state keys=%s | summary=%r | chat_history=%d msg(s):\n%s",
    list(st.session_state.keys()),
    st.session_state.summary,
    len(st.session_state.chat_history),
    "\n".join(
        f"  [{i}] {type(m).__name__}: {m.content[:120]}"
        for i, m in enumerate(st.session_state.chat_history)
    ) or "  (kosong)",
)

st.title(f"🩺 {settings.APP_NAME}")
st.caption("Multi-agent RAG assistant over the Medical Book dataset")

# --- render existing history ---
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- handle new input ---
user_question = st.chat_input("Tanyakan sesuatu tentang topik medis...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Memproses lewat planner -> rag -> evaluator..."):
            result = st.session_state.graph.invoke(
                {
                    "question": user_question,
                    # turns BEFORE this question - it's passed separately above
                    "chat_history": st.session_state.chat_history,
                    "summary": st.session_state.summary,
                }
            )

        answer = result.get("answer", "(tidak ada jawaban)")
        factuality = result.get("factuality")
        tone = result.get("tone")

        if result.get("escalated"):
            st.warning("Respons di-escalate: keyakinan terhadap sumber rendah setelah beberapa percobaan.")

        st.markdown(answer)

        # --- evaluation feedback shown alongside the answer ---
        plan = result.get("plan")
        if plan:
            st.caption("Agen aktif: " + " → ".join(plan))

        if factuality is not None or tone is not None:
            c1, c2 = st.columns(2)
            if factuality is not None:
                c1.caption(f"Factuality: {factuality:.0%}")
            if tone is not None:
                c2.caption(f"Tone: {tone:.0%}")
        if result.get("revised"):
            st.caption("✍️ Jawaban direvisi otomatis oleh Reviser Agent.")
        if result.get("eval_issues"):
            st.caption(f"Catatan Evaluator: {result['eval_issues']}")

        # Summarizer may have updated the running summary this turn.
        if result.get("summary"):
            st.session_state.summary = result["summary"]

    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=answer))

    logger.debug(
        "turn done | chat_history=%d msg(s) | intent=%s | plan=%s | factuality=%s | tone=%s | "
        "revised=%s | escalated=%s | last Q=%r | last A=%r",
        len(st.session_state.chat_history),
        result.get("intent"),
        result.get("plan"),
        factuality,
        tone,
        result.get("revised"),
        result.get("escalated"),
        user_question[:120],
        answer[:120],
    )