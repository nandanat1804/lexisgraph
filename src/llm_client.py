"""
LLM generation stage. API-based by design (per the CPU-friendly brief) -
your laptop just makes an HTTP call instead of hosting a model. Supports
three providers so you can use whichever key you have; switch with
LLM_PROVIDER in .env.
"""
from __future__ import annotations

from .config import config

SYSTEM_PROMPT = """You are a careful legal research assistant. Answer the
user's question using ONLY the provided context passages from legal
documents. Rules:
- Cite the source document and page for every claim, like (source.pdf, p.3)
- If the context does not contain the answer, say so explicitly - never
  guess or use outside knowledge for legal conclusions.
- Be precise about defined terms, section numbers, dates, and amounts.
- Keep the answer concise and structured.
"""


def _build_user_prompt(query: str, context_chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] Source: {c['doc_name']} (page {c['page']})\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )
    return f"""Context passages:
{context_block}

Question: {query}

Answer based only on the context above, with citations."""


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    provider = config.LLM_PROVIDER
    user_prompt = _build_user_prompt(query, context_chunks)

    if provider == "anthropic":
        return _call_anthropic(user_prompt)
    elif provider == "openai":
        return _call_openai(user_prompt)
    elif provider == "groq":
        return _call_groq(user_prompt)
    else:
        return _no_llm_fallback(context_chunks)


def _call_anthropic(user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


def _call_openai(user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
    )
    return resp.choices[0].message.content


def _call_groq(user_prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
    )
    return resp.choices[0].message.content


def _no_llm_fallback(context_chunks: list[dict]) -> str:
    """If no API key is configured, still return something useful:
    the raw top passages, so the retrieval pipeline is testable end to end
    without any API key."""
    lines = ["[No LLM_PROVIDER configured - showing raw retrieved passages]\n"]
    for i, c in enumerate(context_chunks, 1):
        lines.append(f"{i}. ({c['doc_name']}, p.{c['page']}): {c['text'][:300]}...")
    return "\n".join(lines)
