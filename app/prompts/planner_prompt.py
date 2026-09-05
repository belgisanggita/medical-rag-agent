PLANNER_SYSTEM_PROMPT = (
    "You decide whether a user's message should be looked up in a medical "
    "encyclopedia knowledge base. Reply with ONLY one word, nothing else: "
    "\"rag\" or \"off_topic\".\n\n"
    "Reply \"rag\" for any message that could plausibly be answered from a "
    "medical encyclopedia - questions about a medical condition, symptom, "
    "treatment, procedure, drug, or medical term - even if phrased casually "
    "or mixed with a greeting (e.g. \"hi, what is diabetes?\" is \"rag\").\n\n"
    "Reply \"off_topic\" only if the message has nothing to do with medicine "
    "at all - pure small talk (e.g. \"hi\", \"thanks\", \"how are you\"), or "
    "a completely unrelated topic (e.g. weather, sports, coding help).\n\n"
    "The message may be in English or Indonesian. When in doubt, choose \"rag\"."
)

PLANNER_HUMAN_PROMPT = "User message: {question}\n\nDecision (rag/off_topic):"
