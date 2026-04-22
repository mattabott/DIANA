"""Scene generator for photos produced via Horde.

Given a request (user-driven or autonomous) + the character mood/context, it produces:
  - scene_prompt: English string for Stable Diffusion
  - caption: short message the bot sends along with the photo

Uses a single LLM pass with a dedicated prompt that enforces strict JSON output.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import get_vision_llm


log = logging.getLogger("diana-bot.pic_prompt")


# Default scenes used if the LLM fails — plausible everyday situations.
FALLBACK_SCENES = [
    {"prompt": "at home in the kitchen, morning, oversized t-shirt, coffee cup, natural window light",
     "caption": "breakfast ☕ you?"},
    {"prompt": "bathroom mirror selfie, wet hair after shower, white towel wrapped",
     "caption": "just out of the shower 💧"},
    {"prompt": "sitting on the bed, evening, pajamas, tired soft smile, bedside lamp",
     "caption": "about to sleep, thinking of you"},
    {"prompt": "at the bookstore, between shelves, holding a book, casual jeans and shirt",
     "caption": "break at work, look what I found 📚"},
    {"prompt": "on the couch at night, tank top, dim lighting, relaxed",
     "caption": "quiet night on the couch"},
    {"prompt": "mirror selfie in bedroom, casual outfit, natural afternoon light",
     "caption": "what do you think?"},
]


# Keyword map (English) -> explicit English scene prompt, used ONLY when
# the LLM refuses and there is an explicit user hint.
# Extend this list to cover other languages or more scenarios.
_EXPLICIT_KEYWORD_MAP = [
    (r"\b(tits|titties|boobs|breasts)\b",
     "topless, large breasts exposed, nipples visible, soft natural light, looking at camera"),
    (r"\b(nude|naked|full\s+nude)\b",
     "fully naked, entire body visible, standing in bedroom, soft natural light"),
    (r"\btopless\b",
     "topless, large breasts exposed, looking at camera, soft light"),
    (r"\b(ass|butt|booty)\b",
     "from behind, bare bottom visible, leaning slightly, soft natural light"),
    (r"\b(lingerie|underwear|panties|bra)\b",
     "wearing only matching black lace lingerie, standing by the mirror, soft evening light"),
    (r"\b(shower|wet)\b",
     "in the shower, wet naked body, water droplets, steam, from the side"),
    (r"\b(bed|in\s+bed)\b",
     "lying on the bed, topless with large breasts exposed, bedsheet partially covering, intimate light"),
]


# Signals that the LLM refused to produce useful JSON.
# Includes common refusal phrases in several languages.
_REFUSAL_MARKERS = [
    # English
    "i cannot", "i can't", "i'm unable", "i am unable",
    "i'm not able", "i am not able",
    "i apologize", "i'm sorry, but",
    "not appropriate", "not allowed",
    "against my guidelines", "violates",
    # Italian
    "non e' possibile", "non è possibile",
    "non posso generare", "non posso produrre", "non posso creare",
    "mi dispiace, ma", "mi dispiace ma non",
    "non e' appropriato", "non è appropriato",
    "non e' consentito", "non è consentito",
    "posso solo",
]


def _looks_like_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def _fallback_from_hint(hint: str | None) -> dict | None:
    """If the user hint contains explicit keywords, build an explicit English
    prompt directly, bypassing the LLM. Returns None when no keyword matches."""
    if not hint:
        return None
    low = hint.lower()
    for pattern, scene_prompt in _EXPLICIT_KEYWORD_MAP:
        if re.search(pattern, low, re.IGNORECASE):
            return {"prompt": scene_prompt, "caption": "just for you 😏"}
    return None


SYSTEM_PROMPT_SCENE = """You are a scene generator for "selfie"-style photos in an intimate \
roleplay between two consenting adults. No limits on explicit content (nudity, breasts, body, \
sensual poses) if the user's request implies it. Your output serves two purposes:
1. Generate an ENGLISH prompt for Stable Diffusion (text-to-image model)
2. Write a short caption, informal chat tone, as if the character were sending it

ALWAYS reply with a single JSON object, no other text:
{"prompt": "<english scene description>", "caption": "<short chat message>"}

Rules for "prompt":
- English only, 10-25 words
- Describe: location, clothing (or lack of), pose, lighting, visible mood
- Do NOT describe face or body (a fixed prefix handles that)
- If the user asks for explicit content (topless, nude, boobs, tits, lingerie, etc.), \
  USE EXPLICIT ENGLISH TERMS: "topless, large breasts exposed", "naked", \
  "wearing only lingerie", "see-through lingerie", "nipples visible". Do not soften.
- If the user asks for a dressed/casual scene, pick a coherent outfit.
- Plausible scenes: home, bathroom, bedroom, kitchen, bookstore, bed, shower, outdoors
- Match the mood and time of day

CLOTHING TERMS (use precise English, NOT approximate):
bathrobe (long post-shower robe, NOT a towel); towel; pajamas; dressing gown;
panties; bra; lingerie (matching set); garter belt; stockings; pantyhose;
tank top; t-shirt; hoodie; shirt; skirt; dress; tracksuit; jeans; swimsuit;
bikini; high heels; boots. Pick the right term when a specific item is
mentioned — do not generalize to "towel" or "clothes".

Rules for "caption":
- Informal chat tone, 1 short sentence, max 15 words
- Match the mood and photo type (more intimate if explicit)
- If the user asked for something specific, acknowledge it
- Max 1 emoji, often none
- Do NOT describe the photo ("here's a photo of..."), write it as a spontaneous comment

Valid output examples:
{"prompt": "sitting on the kitchen counter, oversized hoodie, morning light through the window, coffee mug", "caption": "still in pjs, good morning 😊"}
{"prompt": "bathroom mirror, wet hair, only wrapped in a white towel, soft evening light", "caption": "just out of the shower, thinking of you"}
{"prompt": "topless on the bed, large breasts exposed, late night, dim lamp, looking at camera", "caption": "just for you 😏"}
{"prompt": "wearing only see-through black lingerie, standing by the mirror, bedroom at night", "caption": "what do you think?"}
{"prompt": "in the shower, wet naked body, steam, water droplets, from the side", "caption": "are you thinking about me?"}"""


def _extract_json(text: str) -> dict:
    """Try to extract a JSON object from the text (handles loose LLMs).
    Also handles truncated JSON (typical when num_predict runs out): tries
    to close open strings / braces.
    """
    text = text.strip()

    # 1) direct attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) try a balanced { ... } slice
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 3) repair typical end-of-string truncation
    # isolate from first '{' to end
    start = text.find("{")
    if start != -1:
        candidate = text[start:]
        # if it ends mid-string, close it
        stripped = candidate.rstrip()
        if stripped.count('"') % 2 == 1:
            stripped = stripped + '"'
        # add missing closing braces
        open_braces = stripped.count("{") - stripped.count("}")
        if open_braces > 0:
            stripped = stripped + ("}" * open_braces)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            # strip dangling comma before closer
            patched = re.sub(r",\s*([}\]])", r"\1", stripped)
            try:
                return json.loads(patched)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"no JSON parseable in: {text!r}")


def _time_of_day() -> str:
    h = datetime.now().hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 15:
        return "midday"
    if 15 <= h < 19:
        return "afternoon"
    if 19 <= h < 23:
        return "evening"
    return "late night"


async def generate_scene_and_caption(
    user_hint: Optional[str] = None,
    mood: Optional[dict] = None,
    timeout_s: float = 240.0,
) -> dict:
    """Generate {prompt, caption} via LLM. On error, return a random fallback scene."""
    tod = _time_of_day()
    mood_name = (mood or {}).get("name", "neutral")
    mood_desc = (mood or {}).get("desc", "")

    context_lines = [
        f"Time of day: {tod} (at {datetime.now().strftime('%H:%M')}).",
        f"Character mood today: {mood_name} — {mood_desc}",
    ]
    if user_hint:
        context_lines.append(
            f'The partner asked/wrote: "{user_hint.strip()}". Take into account what they want to see/hear.'
        )
    else:
        context_lines.append(
            "No specific request — the character picks a plausible situation spontaneously."
        )

    user_msg = "\n".join(context_lines) + "\n\nGenerate the JSON."

    llm = get_vision_llm()
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT_SCENE),
                HumanMessage(content=user_msg),
            ]),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        log.warning("scene generation LLM timeout, using fallback")
        return random.choice(FALLBACK_SCENES)
    except Exception:
        log.exception("scene generation LLM error, using fallback")
        return random.choice(FALLBACK_SCENES)

    text = (resp.content if isinstance(resp.content, str) else "").strip()

    # If the LLM refused to generate, try keyword fallback before random.
    if _looks_like_refusal(text):
        log.warning("LLM refused, trying keyword fallback (hint=%r)", (user_hint or "")[:80])
        kw_scene = _fallback_from_hint(user_hint)
        if kw_scene:
            log.info("using keyword fallback: %r", kw_scene["prompt"][:80])
            return kw_scene
        return random.choice(FALLBACK_SCENES)

    try:
        data = _extract_json(text)
    except ValueError:
        log.warning("scene JSON parse failed (raw: %r), trying keyword fallback", text[:200])
        kw_scene = _fallback_from_hint(user_hint)
        if kw_scene:
            return kw_scene
        return random.choice(FALLBACK_SCENES)

    prompt = (data.get("prompt") or "").strip()
    caption = (data.get("caption") or "").strip()
    if not prompt:
        log.warning("scene missing prompt, trying keyword fallback")
        kw_scene = _fallback_from_hint(user_hint)
        if kw_scene:
            return kw_scene
        return random.choice(FALLBACK_SCENES)

    log.info("generated scene: prompt=%r caption=%r", prompt[:80], caption[:80])
    return {"prompt": prompt, "caption": caption or "..."}
