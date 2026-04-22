"""ReAct agent with memory (history + facts) injected into the context.

Exposes:
  - `generate_reply`: ReAct-based text reply
  - `generate_vision_reply`: direct LLM reply for photo comments (no ReAct)
  - `extract_facts`: second-pass LLM to pull personal facts from user text
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import create_react_agent

from src.config import CONFIG
from src.tools import ALL_TOOLS
from src.persona import get_persona_system_prompt


log = logging.getLogger("diana-bot.agent")


TOOLS_NOTE = """\n\nAVAILABLE TOOLS (use sparingly, only when truly needed):
- get_current_datetime: only if asked for date/time or when proposing specific times
- remember_this: only when he shares long-term personal info
  (name, job, strong preferences, dates, important people). Don't save small talk.

In all other cases reply directly without calling tools."""


_agent: Any = None
_llm_vision: ChatOllama | None = None


def get_agent() -> Any:
    global _agent
    if _agent is None:
        llm = ChatOllama(
            model=CONFIG.ollama_model,
            base_url=CONFIG.ollama_host,
            reasoning=False,
            temperature=0.8,
            num_ctx=8192,
            num_predict=250,
            keep_alive=CONFIG.ollama_keep_alive,
        )
        _agent = create_react_agent(model=llm, tools=ALL_TOOLS)
        log.info(
            "ReAct agent ready: model=%s tools=%s",
            CONFIG.ollama_model,
            [t.name for t in ALL_TOOLS],
        )
    return _agent


def get_vision_llm() -> ChatOllama:
    """Direct LLM (no ReAct) for the vision path: avoids the doubled latency
    that ReAct would add on top of an already heavy image payload.
    """
    global _llm_vision
    if _llm_vision is None:
        _llm_vision = ChatOllama(
            model=CONFIG.ollama_model,
            base_url=CONFIG.ollama_host,
            reasoning=False,
            temperature=0.8,
            num_ctx=4096,
            num_predict=180,
            keep_alive=CONFIG.ollama_keep_alive,
        )
        log.info("Vision LLM ready (direct, no ReAct)")
    return _llm_vision


def _format_event_line(ev: dict) -> str:
    ts = ev.get("created_at", "")
    # created_at is ISO "YYYY-MM-DDTHH:MM:SS"; render short month+day+time
    try:
        from datetime import datetime as _dt
        d = _dt.fromisoformat(ts)
        label = d.strftime("%d %b %H:%M")
    except Exception:
        label = ts[:16]
    return f"[{label}] {ev.get('text','').strip()}"


def _build_system_prompt(
    facts: list[str],
    summary: str | None,
    with_tools: bool = False,
    daily_mood: dict | None = None,
    intimacy_count: int | None = None,
    events: list[dict] | None = None,
) -> str:
    parts = [
        get_persona_system_prompt(
            daily_mood=daily_mood,
            intimacy_count=intimacy_count,
        )
    ]
    if with_tools:
        parts.append(TOOLS_NOTE)
    if facts:
        parts.append(
            "\nTHINGS YOU KNOW ABOUT HIM (always true):\n- "
            + "\n- ".join(facts)
        )
    if events:
        lines = [_format_event_line(e) for e in events]
        parts.append(
            "\nRECENT EVENTS IN HIS LIFE (told outside chat in the last days):\n- "
            + "\n- ".join(lines)
        )
    if summary:
        parts.append(
            "\nSUMMARY OF PREVIOUS CONVERSATIONS:\n" + summary
        )
    return "\n".join(parts)


def _history_to_messages(history: list[dict]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for row in history:
        if row["role"] == "user":
            out.append(HumanMessage(content=row["content"]))
        elif row["role"] == "assistant":
            out.append(AIMessage(content=row["content"]))
    return out


async def generate_reply(
    user_text: str,
    history: list[dict] | None = None,
    facts: list[str] | None = None,
    summary: str | None = None,
    daily_mood: dict | None = None,
    intimacy_count: int | None = None,
    events: list[dict] | None = None,
) -> str:
    """Text reply via ReAct agent (tool calling + memory)."""
    agent = get_agent()
    system = _build_system_prompt(
        facts or [], summary, with_tools=True,
        daily_mood=daily_mood, intimacy_count=intimacy_count,
        events=events,
    )
    msgs: list[BaseMessage] = [SystemMessage(content=system)]
    msgs.extend(_history_to_messages(history or []))
    msgs.append(HumanMessage(content=user_text))

    log.debug("Agent request: msgs=%d user=%r", len(msgs), user_text)
    result = await agent.ainvoke({"messages": msgs})
    messages = result.get("messages", [])

    final_text = ""
    tool_calls_used: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", []) or []:
                tool_calls_used.append(tc.get("name", "?"))
            content = m.content if isinstance(m.content, str) else ""
            if content.strip():
                final_text = content.strip()
    if tool_calls_used:
        log.info("Agent used tools: %s", tool_calls_used)
    log.info("Agent reply (%d chars): %s", len(final_text), final_text[:120])
    return final_text or "..."


FACT_EXTRACTION_SYSTEM = """You are a fact extractor. Analyze the user's message \
and identify ONLY important long-term personal information about the sender:
- name, age, job, where they live
- important people (children, partner, friends), pets, relationships
- strong durable preferences (music, food, sports, favorite books)
- important dates, anniversaries
- life projects, milestones, significant events

IGNORE small talk, passing comments, fleeting opinions, questions, greetings, mood-of-the-moment.

Reply with ONE FACT PER LINE, short and clear, in third person singular \
("His name is Matt", "He's 34", "He has a son", "He works in AI").

If the message contains NO relevant personal facts, reply exactly with: NONE

No other explanation, no bullet points, just the list or NONE."""


_llm_extract: ChatOllama | None = None


def _get_extract_llm() -> ChatOllama:
    """Minimal dedicated LLM for fact extraction (low num_predict for speed)."""
    global _llm_extract
    if _llm_extract is None:
        _llm_extract = ChatOllama(
            model=CONFIG.ollama_model,
            base_url=CONFIG.ollama_host,
            reasoning=False,
            temperature=0.2,
            num_ctx=2048,
            num_predict=120,
            keep_alive=CONFIG.ollama_keep_alive,
        )
    return _llm_extract


async def extract_facts(user_text: str, timeout_s: float = 90.0) -> list[str]:
    """Minimal LLM pass that extracts personal facts from a message.
    Internal timeout: if the LLM takes longer than `timeout_s`, returns [].
    """
    if len(user_text.strip()) < 15:
        return []
    llm = _get_extract_llm()
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=FACT_EXTRACTION_SYSTEM),
                HumanMessage(content=user_text),
            ]),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        log.warning("fact extraction timeout after %.0fs (msg: %r)", timeout_s, user_text[:60])
        return []
    except Exception:
        log.exception("fact extraction LLM call failed")
        return []
    text = (resp.content if isinstance(resp.content, str) else "").strip()
    if not text or text.upper().startswith("NONE"):
        return []
    # Split by line and clean up.
    facts: list[str] = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-*•").strip()
        if not line or line.upper() == "NONE":
            continue
        if len(line) < 5 or len(line) > 200:
            continue
        facts.append(line)
    return facts


async def generate_vision_reply(
    caption: str,
    image_b64: str,
    facts: list[str] | None = None,
    daily_mood: dict | None = None,
    intimacy_count: int | None = None,
    events: list[dict] | None = None,
) -> str:
    """Photo reply: direct path (no ReAct) to keep latency manageable.

    Does not use history (photos are atomic events) nor tool calls.
    Includes facts for persona coherence, but keeps context minimal so the
    image tokens have room.
    """
    llm = get_vision_llm()
    system = _build_system_prompt(
        facts or [], summary=None,
        daily_mood=daily_mood, intimacy_count=intimacy_count,
        events=events,
    )

    text_prompt = caption.strip() if caption else "sending you a photo, tell me what you think"
    msgs: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(
            content=[
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ]
        ),
    ]
    log.debug("Vision request: caption=%r img_b64_len=%d", caption, len(image_b64))
    resp = await llm.ainvoke(msgs)
    text = (resp.content if isinstance(resp.content, str) else "").strip()
    log.info("Vision reply (%d chars): %s", len(text), text[:120])
    return text or "..."
