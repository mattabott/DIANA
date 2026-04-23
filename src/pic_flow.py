"""Generated photo flow: used by manual /pic, reactive requests, and autonomous ticks.

Centralizes:
  - rate limit / daily quota
  - scene generation (English prompt + caption)
  - Horde call
  - photo send + DB log

Shared by bot.py (manual+reactive) and scheduler.py (autonomous).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from telegram import Bot
from telegram.constants import ChatAction

from src.config import CONFIG
from src.memory import AsyncMemory
from src.horde import generate_image, HordeError, CensorshipDetected
from src.persona import (
    AVAILABLE_STYLES, DEFAULT_STYLE, get_style_config,
    get_visual_prompt_prefix, pick_visual_shot_type,
)
from src.pic_prompt import generate_scene_and_caption


SETTING_STYLE_KEY = "pic_style"


async def get_active_style(amem: AsyncMemory) -> str:
    """Read the active style from the DB with default fallback + validation."""
    cur = await amem.get_setting(SETTING_STYLE_KEY, DEFAULT_STYLE)
    return cur if cur in AVAILABLE_STYLES else DEFAULT_STYLE


log = logging.getLogger("diana-bot.pic_flow")


# Regex patterns for detecting a photo request in chat.
# Intentionally loose: false positives are preferable to missed triggers.
# (These are English-language patterns. Add/translate for other languages
# by editing this list.)
PHOTO_REQUEST_PATTERNS = [
    r"\bsend\s+me\s+.{0,30}?(photo|picture|pic|selfie|snap)\b",
    r"\bgimme\s+.{0,15}?(photo|picture|pic|selfie)\b",
    r"\bshow\s+me\s+.{0,30}?(photo|picture|pic|selfie|you|what|how|your)\b",
    r"\bi\s+(want|need|'d\s+like)\s+.{0,25}?(photo|picture|pic|selfie)\b",
    r"\bcan\s+you\s+send\s+.{0,25}?(photo|picture|pic|selfie)\b",
    r"\bcould\s+you\s+send\s+.{0,25}?(photo|picture|pic|selfie)\b",
    r"\ba\s+selfie\b",
    r"\b(your|a)\s+(photo|picture|pic|selfie)\b",
    r"\b(photo|pic)\s+of\s+you\b",
    r"\bpic\s+(please|pls)\b",
    r"\bwould\s+love\s+.{0,20}?(photo|picture|pic|selfie)\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PHOTO_REQUEST_PATTERNS]


# Words indicating the ASSISTANT (bot) just mentioned a photo.
_PHOTO_CONTEXT_RE = re.compile(
    r"\b(photo|picture|pic|selfie|snap|image|shot|portrait)\b",
    re.IGNORECASE,
)

# Affirmative/insistent words that in a follow-up imply "yes, send it":
# used together with a photo context in the bot's last message.
_PHOTO_AFFIRMATIVE_RE = re.compile(
    r"\b("
    r"please|pls|go\s+on|come\s+on|do\s+it|yes\s+please"
    r"|send\s+one|take\s+one|snap\s+one|gimme\s+one"
    r"|go\s+ahead|ok\s+go|yes\s+go|yeah"
    r"|a\s+good\s+one|pretty\s+please"
    r")\b",
    re.IGNORECASE,
)


def is_photo_request(text: str) -> bool:
    """True if the user message looks like a standalone photo request."""
    if not text:
        return False
    low = text.lower()
    return any(p.search(low) for p in _COMPILED_PATTERNS)


def is_photo_continuation(user_text: str, last_assistant_text: str) -> bool:
    """Hybrid context detector: the request is implicit but the conversation
    was talking about photos. True when:
      - the bot's last message contains photo-related words, AND
      - the new user message contains insistence/affirmation.
    """
    if not user_text or not last_assistant_text:
        return False
    if not _PHOTO_CONTEXT_RE.search(last_assistant_text):
        return False
    return bool(_PHOTO_AFFIRMATIVE_RE.search(user_text))


async def _keepalive_action(bot: Bot, chat_id: int, action: str, stop: asyncio.Event) -> None:
    """Like the typing keepalive but with configurable action (e.g., UPLOAD_PHOTO)."""
    try:
        while not stop.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=action)
            except Exception as exc:
                log.debug("chat_action failed: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        pass


async def can_send_pic(amem: AsyncMemory) -> tuple[bool, str]:
    """Return (True, '') if it's ok to generate a photo now, else (False, reason)."""
    count = await amem.pic_count_today()
    if count >= CONFIG.pic_max_per_day:
        return False, f"daily quota {count}/{CONFIG.pic_max_per_day}"
    last = await amem.last_pic_at()
    if last is not None:
        elapsed = (datetime.now() - last).total_seconds()
        if elapsed < CONFIG.pic_cooldown_min * 60:
            left = int(CONFIG.pic_cooldown_min * 60 - elapsed)
            return False, f"cooldown {left}s left"
    return True, ""


def _compose_full_prompt(scene_prompt: str, style: str = DEFAULT_STYLE) -> str:
    prefix = get_visual_prompt_prefix(style)
    shot = pick_visual_shot_type(style)
    parts = [p for p in (prefix, shot, scene_prompt) if p]
    return ", ".join(parts)


async def run_pic_flow(
    bot: Bot,
    amem: AsyncMemory,
    chat_id: int,
    trigger_type: str,           # 'manual' | 'reactive' | 'autonomous'
    user_hint: Optional[str] = None,     # user request text (reactive) or None
    explicit_prompt: Optional[str] = None,   # manual /pic: bypass LLM and use this
    mood: Optional[dict] = None,
    status_message_text: Optional[str] = None,
) -> Optional[str]:
    """End-to-end: generate + send a photo. Handles quota, rate-limit, logging.
    Returns the sent caption, or None if nothing was generated (quota/error).
    """
    ok, reason = await can_send_pic(amem)
    if not ok:
        log.info("pic flow skip (%s): %s", trigger_type, reason)
        return None

    # 1. Decide scene_prompt + caption
    if explicit_prompt:
        scene_prompt = explicit_prompt
        caption = f"/pic: {explicit_prompt[:200]}"
    else:
        scene = await generate_scene_and_caption(user_hint=user_hint, mood=mood)
        scene_prompt = scene["prompt"]
        caption = scene["caption"]

    style = await get_active_style(amem)
    style_cfg = get_style_config(style)
    full_prompt = _compose_full_prompt(scene_prompt, style)
    log.info("pic flow %s [style=%s]: scene=%r caption=%r",
             trigger_type, style, scene_prompt[:80], caption[:80])

    # 2. Initial status + typing/upload_photo indicator
    status_msg = None
    if status_message_text:
        try:
            status_msg = await bot.send_message(chat_id=chat_id, text=status_message_text)
        except Exception:
            log.exception("status message send failed")

    stop = asyncio.Event()
    action_task = asyncio.create_task(
        _keepalive_action(bot, chat_id, ChatAction.UPLOAD_PHOTO, stop)
    )

    try:
        img_bytes = await generate_image(
            full_prompt,
            models=style_cfg["models"] or None,
            negative_override=style_cfg["negative_prompt"] or None,
        )
    except CensorshipDetected as e:
        log.warning("censored by Horde filter: %s", e)
        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:
                pass
        # In-character message instead of the censorship banner.
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="mmh it came out wrong, the system blocked it. Try a different scene?",
            )
        except Exception:
            pass
        return None
    except HordeError as e:
        log.error("horde error in pic flow: %s", e)
        if status_msg is not None:
            try:
                await status_msg.edit_text(f"❌ generation error: {e}")
            except Exception:
                pass
        return None
    except Exception:
        log.exception("unexpected error in pic flow")
        if status_msg is not None:
            try:
                await status_msg.edit_text("❌ unexpected error during generation")
            except Exception:
                pass
        return None
    finally:
        stop.set()
        await action_task

    # 3. Send photo + caption (Telegram caption or separate message)
    try:
        # Delete the status message before sending (cosmetic).
        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:
                pass
        await bot.send_photo(chat_id=chat_id, photo=img_bytes, caption=caption[:1024])
    except Exception:
        log.exception("send_photo failed")
        return None

    # 4. DB log + assistant-role message in history.
    # IMPORTANT: we save ONLY the caption to history (the same text the user
    # saw below the photo). If we saved the scene prompt as well (e.g.
    # "[photo sent: <prompt>] ..."), the LLM would learn that pattern from
    # past messages and imitate it as a text reply instead of actually
    # generating a photo. The scene is still kept in pic_log for debugging.
    await amem.log_pic(trigger_type, scene_prompt, caption)
    await amem.save_message("assistant", caption)

    return caption
