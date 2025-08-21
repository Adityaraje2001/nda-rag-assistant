# src/llm_provider.py
from __future__ import annotations

import os
import time
import json
from typing import Optional

FALLBACK_TEXT = "Not found in the provided documents.\n\nSources: "

# Default to Gemma 2 9B Instruct (free) via OpenRouter
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY")                          # e.g., "sk-or-..."
ROUTER_BASE_URL = os.getenv("ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ROUTER_MODEL   = os.getenv("ROUTER_MODEL", "google/gemma-2-9b-it:free")

def _env_ok() -> bool:
    return bool(ROUTER_API_KEY and ROUTER_BASE_URL and ROUTER_MODEL)

def _dbg(model_override: Optional[str] = None):
    print("LLM DEBUG using:", model_override or ROUTER_MODEL, "base:", ROUTER_BASE_URL, "key?", bool(ROUTER_API_KEY))

def llm_call(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """
    Call OpenRouter (OpenAI-compatible) Chat Completions API with Gemma 2 9B Instruct (free by default).
    Returns model text on success; safe fallback text on failure (after logging the error).
    """
    m = model or ROUTER_MODEL
    _dbg(m)
    if not _env_ok():
        return FALLBACK_TEXT

    # 1) Preferred path: OpenAI SDK (requires `pip install openai>=1.0`)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=ROUTER_API_KEY, base_url=ROUTER_BASE_URL)

        for i, delay in enumerate([0.0, 0.8, 1.6]):
            if delay:
                time.sleep(delay)
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a careful legal assistant. Follow instructions strictly. Answer only from the provided context when given."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=max(0.0, min(2.0, float(temperature))),
                    max_tokens=int(max_tokens),
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
            except Exception as e:
                import traceback
                print(f"LLM SDK TRY {i+1} ERROR:", repr(e))
                traceback.print_exc()
        # Fall through to HTTP if SDK attempts fail
    except Exception as e:
        print("LLM SDK IMPORT/INIT ERROR:", repr(e))

    # 2) HTTP fallback (requires `pip install requests`)
    try:
        import requests

        url = ROUTER_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {ROUTER_API_KEY}",
            "Content-Type": "application/json",
            # Optional attribution headers recommended by OpenRouter:
            # "HTTP-Referer": "http://localhost:7860",
            # "X-Title": "Local RAG",
        }
        payload = {
            "model": m,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful legal assistant. Follow instructions strictly. Answer only from the provided context when given."
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": max(0.0, min(2.0, float(temperature))),
            "max_tokens": int(max_tokens),
        }

        for i, delay in enumerate([0.0, 0.8, 1.6]):
            if delay:
                time.sleep(delay)
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            if r.status_code >= 400:
                # Print server error body to diagnose (401/402/403/404/429/5xx/400)
                print(f"LLM HTTP TRY {i+1} ERROR:", r.status_code, r.text)
                continue
            try:
                data = r.json()
            except Exception:
                print("LLM HTTP JSON ERROR: raw:", r.text[:1000])
                continue
            try:
                content = (data["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                print("LLM HTTP PARSE ERROR: payload:", json.dumps(data)[:1000])
                content = ""
            if content:
                return content

    except Exception as e:
        import traceback
        print("LLM HTTP FALLBACK EXCEPTION:", repr(e))
        traceback.print_exc()

    # 3) Final fallback
    return FALLBACK_TEXT
