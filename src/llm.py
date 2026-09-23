"""共享 LLM 客户端：DeepSeek（OpenAI 兼容），无 key 时返回 None 走降级。"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def build_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY", "no-key"),
        temperature=0,
    )


LLM_AVAILABLE = bool(os.getenv("DEEPSEEK_API_KEY"))
llm = build_llm() if LLM_AVAILABLE else None
