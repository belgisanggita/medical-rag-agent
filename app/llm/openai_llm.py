"""
Shared ChatOpenAI factory + invocation helper for the medical RAG pipeline,
targeting OpenRouter's OpenAI-compatible API. All agents (planner, rag,
evaluator, summarizer) build their prompts with langchain_core.prompts and
call the LLM through invoke_prompt() below, so credentials/model config
live in one place and every request/response is logged the same way.

The configured model (openai/gpt-oss-120b) is a reasoning model - it always
spends part of its token budget "thinking" before the actual answer, and
OpenRouter rejects disabling that ("Reasoning is mandatory for this
endpoint" when passing reasoning.enabled=false). reasoning_max_tokens caps
that internal budget via OpenRouter's `reasoning` extension (passed through
extra_body, since it's not a standard OpenAI param) so the rest of
max_tokens is left for the actual answer - without this cap, a small
max_tokens (e.g. the Evaluator's short numeric answer) gets consumed
entirely by reasoning and the response comes back empty.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import properties_setup as settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_llm(temperature: float = 0.3, max_tokens: int = 1024, reasoning_max_tokens: int = 300) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENROUTER_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"max_tokens": reasoning_max_tokens}},
    )


def invoke_prompt(
    prompt: ChatPromptTemplate,
    inputs: dict,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    reasoning_max_tokens: int = 300,
) -> str:
    """
    Formats `prompt` with `inputs` (resolving any MessagesPlaceholder, e.g.
    chat_history, into real messages), logs the exact message list sent to
    the LLM and the raw response, then returns the response text. Every
    agent (planner, rag, evaluator, summarizer) goes through this one
    function so every LLM call is logged the same way.
    """
    messages = prompt.format_messages(**inputs)
    logger.info("LLM request:\n" + "\n".join(f"[{m.type}] {m.content}" for m in messages))

    llm = get_llm(temperature=temperature, max_tokens=max_tokens, reasoning_max_tokens=reasoning_max_tokens)
    response = llm.invoke(messages)

    logger.info(f"LLM response: {response.content!r}")

    if not response.content:
        logger.warning(f"LLM returned no content (likely truncated by max_tokens={max_tokens})")
        return ""
    return response.content
