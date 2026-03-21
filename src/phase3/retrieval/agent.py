import os
from openai import OpenAI

# ── DeepSeek 客户端 ────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY",    "sk-c4ee6d08aeab4aa5a32622fee0a69e0d")
DEEPSEEK_BASE_URL   = os.getenv("DEEPSEEK_BASE_URL",   "https://api.deepseek.com")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def call_llm(messages: list[dict]) -> str:
    response = _client.chat.completions.create(
        model=DEEPSEEK_MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""