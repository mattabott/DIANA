"""Autonomous message logic.

Every `check_interval_min` minutes the job calls `maybe_send_autonomous`:
  - checks allowed hours and daily rate limit
  - checks cooldown since last interaction
  - generates a message with the LLM (dedicated prompt, no ReAct)
  - sends it via Telegram and logs it in the DB
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Awaitable, Optional, TypeVar

from telegram import Bot
from telegram.constants import ChatAction
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import CONFIG
from src.memory import AsyncMemory
from src.agent import get_vision_llm
from src.persona import get_persona_system_prompt, ensure_today_mood
from src.pic_flow import run_pic_flow, can_send_pic


T = TypeVar("T")


log = logging.getLogger("diana-bot.scheduler")

# Minimum cooldown since last interaction (either direction) before an
# autonomous message can be sent. Avoids sounding obsessive.
MIN_COOLDOWN_MIN = 90


AUTONOMOUS_INSTRUCTION = """You want to send a spontaneous message to your partner \
(out of the blue, the way you do when someone is on your mind). Think of a \
plausible hook: a thought, a curiosity, a photo you just scrolled past, a \
memory, a casual question, something that happened today.

Rules:
- ONE message only, 1-2 sentences, short
- Do NOT start with "hi" or "hey" (you're already in conversation)
- Informal, natural tone, like a real chat
- Max 1 emoji
- NO meta-commentary: reply ONLY with the message to send, no quotes, no explanations.
"""


def _in_allowed_hours(now: datetime) -> bool:
    lo = CONFIG.autonomous_min_hour
    hi = CONFIG.autonomous_max_hour
    return lo <= now.hour < hi


async def _keepalive_typing(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    """Refreshes 'typing...' every 4s until `stop` is set."""
    try:
        while not stop.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception as exc:
                log.debug("typing action failed: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        pass


async def _with_typing(bot: Bot, chat_id: int, coro: Awaitable[T]) -> T:
    """Run a coroutine while keeping the typing indicator alive."""
    stop = asyncio.Event()
    task = asyncio.create_task(_keepalive_typing(bot, chat_id, stop))
    try:
        return await coro
    finally:
        stop.set()
        await task


async def _generate_autonomous_text(facts: list[str], recent: list[dict]) -> str:
    """Use the vision LLM (direct, no ReAct) to generate the message.
    Rationale: skipping ReAct avoids doubling latency and unneeded tool calls here.
    """
    llm = get_vision_llm()
    context_bits: list[str] = []
    if facts:
        context_bits.append("Things you know about him:\n- " + "\n- ".join(facts))
    if recent:
        lines = [f"{m['role']}: {m['content']}" for m in recent[-5:]]
        context_bits.append("Latest chat exchanges:\n" + "\n".join(lines))

    system_text = get_persona_system_prompt()
    if context_bits:
        system_text += "\n\n" + "\n\n".join(context_bits)

    msgs = [
        SystemMessage(content=system_text),
        HumanMessage(content=AUTONOMOUS_INSTRUCTION),
    ]
    resp = await llm.ainvoke(msgs)
    text = (resp.content if isinstance(resp.content, str) else "").strip()
    # Strip surrounding quotes if the model added them.
    if len(text) >= 2 and text[0] in "\"'«" and text[-1] in "\"'»":
        text = text[1:-1].strip()
    return text


async def maybe_send_autonomous(bot: Bot, amem: AsyncMemory) -> Optional[str]:
    """Decide whether to send an autonomous message now and send it if ok.
    Returns the sent text, or None if skipped (reason at log.debug).
    """
    if not CONFIG.autonomous_enabled:
        log.debug("skip: autonomous disabled")
        return None

    now = datetime.now()
    if not _in_allowed_hours(now):
        log.debug("skip: out of hours (%d)", now.hour)
        return None
    if not CONFIG.autonomous_weekend and now.weekday() >= 5:
        log.debug("skip: weekend disabled (weekday=%d)", now.weekday())
        return None

    sent_today = await amem.autonomous_count_today()
    if sent_today >= CONFIG.autonomous_max_per_day:
        log.debug("skip: daily cap reached (%d)", sent_today)
        return None

    last = await amem.last_any_message_at()
    if last is not None:
        elapsed = now - last
        if elapsed < timedelta(minutes=MIN_COOLDOWN_MIN):
            log.debug("skip: cooldown (%s since last)", elapsed)
            return None

    # Dice roll: even when checks pass, don't always send. Adds natural variety.
    # Probability grows with the wait: 1h = 20%, 3h+ = 80%.
    if last is not None:
        hours = (now - last).total_seconds() / 3600.0
    else:
        hours = 24.0
    prob = max(0.15, min(0.8, 0.15 + hours * 0.15))
    roll = random.random()
    if roll > prob:
        log.debug("skip: dice roll %.2f > %.2f (hours=%.1f)", roll, prob, hours)
        return None

    # Dice roll: with probability PIC_AUTONOMOUS_PROB%, send a photo instead of text.
    if random.randint(1, 100) <= CONFIG.pic_autonomous_prob:
        pic_ok, pic_reason = await can_send_pic(amem)
        if pic_ok:
            log.info("autonomous: choosing photo path")
            mood = await ensure_today_mood(amem)
            caption = await run_pic_flow(
                bot=bot,
                amem=amem,
                chat_id=CONFIG.allowed_chat_id,
                trigger_type="autonomous",
                user_hint=None,
                mood=mood,
            )
            if caption is not None:
                # log_autonomous tracks the daily autonomous-message quota.
                await amem.log_autonomous(f"[autonomous photo] {caption}")
                return caption
            log.info("autonomous photo failed, falling through to text")
        else:
            log.debug("autonomous photo skip: %s - fallback to text", pic_reason)

    # Standard text path.
    facts = await amem.list_facts(limit=50)
    recent = await amem.recent_messages(limit=6)
    log.info("autonomous: generating text... facts=%d recent=%d", len(facts), len(recent))
    text = await _with_typing(
        bot, CONFIG.allowed_chat_id, _generate_autonomous_text(facts, recent)
    )
    if not text:
        log.warning("autonomous: empty text")
        return None

    await bot.send_message(chat_id=CONFIG.allowed_chat_id, text=text)
    await amem.log_autonomous(text)
    log.info("autonomous sent (%d chars): %s", len(text), text[:100])
    return text


async def force_send_autonomous(bot: Bot, amem: AsyncMemory) -> str:
    """Variant used by /ping: bypasses checks/rate-limits for testing."""
    facts = await amem.list_facts(limit=50)
    recent = await amem.recent_messages(limit=6)
    text = await _with_typing(
        bot, CONFIG.allowed_chat_id, _generate_autonomous_text(facts, recent)
    )
    if not text:
        text = "..."
    await bot.send_message(chat_id=CONFIG.allowed_chat_id, text=text)
    await amem.log_autonomous(text)
    log.info("autonomous FORCED (%d chars): %s", len(text), text[:100])
    return text
