EVALUATOR_SYSTEM_PROMPT = (
    "You are an evaluator that judges whether an answer is truly grounded in "
    "the given context (not hallucinated) and relevant to the question. "
    "Reply with ONLY a single decimal number between 0 and 1, nothing else: "
    "1 means the answer is fully supported by the context and answers the "
    "question well, 0 means the answer is not supported by the context at all "
    "or is made up. The context, question and answer may be in English or "
    "Indonesian - judge the meaning regardless of language."
)

EVALUATOR_HUMAN_PROMPT = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nScore (0-1):"
