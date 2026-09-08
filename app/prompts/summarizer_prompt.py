SUMMARIZER_SYSTEM_PROMPT = (
    "You summarize the conversation between a user and a medical assistant "
    "into a short running summary (at most 5 sentences), focusing on the "
    "topics discussed and any information relevant to continuing the "
    "conversation. "
    "Language rule, in priority order: (1) if the user's most recent message "
    "asks for a specific language - e.g. 'ringkas dalam bahasa Indonesia', "
    "'pakai bahasa Indonesia', 'summarize in English' - write the ENTIRE "
    "summary in that language, translating earlier content as needed; "
    "(2) otherwise, write it in the language the user has mostly been using "
    "(English or Indonesian). Never mix languages in the summary."
)

SUMMARIZER_PREVIOUS_SUMMARY_PROMPT = "Previous summary: {previous_summary}"

SUMMARIZER_HUMAN_PROMPT = (
    "Update the running summary now. Follow the language rule above, giving "
    "priority to any explicit language request in my latest message."
)
