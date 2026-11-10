"""Grok (xAI) API 最小客户端 — 零额外依赖（只用 stdlib + requests）。

配置从环境变量读取（见仓库根目录的 .env 文件）：
    XAI_API_KEY   你的 API Key，来自 console.x.ai（以 "xai-" 开头）
    XAI_BASE_URL  默认 https://api.x.ai/v1
    XAI_MODEL     默认 grok-4.6

用法：
    from grok_client import chat

    print(chat("用一句话介绍你自己"))
"""
import os
from pathlib import Path

import requests

# ---- 极简 .env 加载器（避免引入 python-dotenv 依赖）----
def _load_dotenv(path: Path | None = None) -> None:
    path = path or (Path(__file__).resolve().parent / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()

BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
MODEL = os.getenv("XAI_MODEL", "grok-4.6")
API_KEY = os.getenv("XAI_API_KEY", "")


def _assert_key(api_key: str) -> None:
    if not api_key or api_key.startswith("xai-在这里"):
        raise RuntimeError("XAI_API_KEY 未配置：请在仓库根目录的 .env 中填入真实 Key。")


def chat(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> str:
    """一次性对话，返回助手回复文本。"""
    key = api_key or API_KEY
    _assert_key(key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{base_url or BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model or MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


class Grok:
    """带对话上下文的客户端，可连续多轮提问。"""

    def __init__(
        self,
        model: str | None = None,
        system: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or MODEL
        self.base_url = base_url or BASE_URL
        self.api_key = api_key or API_KEY
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def chat(self, prompt: str, **kwargs) -> str:
        _assert_key(self.api_key)
        self.messages.append({"role": "user", "content": prompt})
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "messages": self.messages, **kwargs},
            timeout=120,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        return reply


if __name__ == "__main__":
    print(chat("用中文说一句你好"))
