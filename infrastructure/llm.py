from langchain_mistralai import ChatMistralAI

from infrastructure.config import MISTRAL_API_KEY, MISTRAL_MODEL


def get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=0,
    )
