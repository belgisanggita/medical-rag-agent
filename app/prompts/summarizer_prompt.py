SUMMARIZER_SYSTEM_PROMPT = (
    "You summarize the conversation between a user and a medical assistant "
    "into a short running summary (at most 5 sentences), focusing on the "
    "topics discussed and any information relevant to continuing the "
    "conversation. Write the summary in the same language the user has been "
    "using in the conversation (English or Indonesian)."
)

SUMMARIZER_PREVIOUS_SUMMARY_PROMPT = "Previous summary: {previous_summary}"

SUMMARIZER_HUMAN_PROMPT = "Update the summary:"
