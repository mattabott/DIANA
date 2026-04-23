"""Loads the persona from the YAML file and builds the system prompt.

Features: scenario framing, explicit mode, variable daily mood, growing
intimacy level, visual prompt prefix for image generation.
"""
from __future__ import annotations

import logging
import random
import yaml
from pathlib import Path
from typing import Any, Optional

from src.config import CONFIG


log = logging.getLogger("diana-bot.persona")


def _load_yaml() -> dict[str, Any]:
    path: Path = CONFIG.persona_file
    if not path.exists():
        log.warning("persona file not found at %s, using minimal defaults", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _bullet(lines: list[str]) -> str:
    return "\n".join(f"- {x}" for x in lines)


_CACHED_DATA: dict[str, Any] | None = None


def _data(force_reload: bool = False) -> dict[str, Any]:
    global _CACHED_DATA
    if _CACHED_DATA is None or force_reload:
        _CACHED_DATA = _load_yaml()
        log.info("persona yaml loaded (%d top-level keys)", len(_CACHED_DATA))
    return _CACHED_DATA


def pick_daily_mood() -> dict[str, str]:
    """Randomly pick a mood from those defined in the YAML."""
    moods = _data().get("daily_moods") or []
    if not moods:
        return {"name": "neutral", "desc": "standard tone"}
    choice = random.choice(moods)
    return {"name": choice.get("name", "neutral"), "desc": choice.get("desc", "")}


def intimacy_level_for(interaction_count: int) -> int:
    """Simple formula: +1 level every 20 interactions, capped at 10."""
    return max(1, min(10, 1 + interaction_count // 20))


def intimacy_description(level: int) -> str:
    levels = _data().get("intimacy_levels") or {}
    # YAML loads numeric keys as int
    return levels.get(level) or levels.get(str(level)) or ""


def _build_persona_base(p: dict[str, Any]) -> str:
    ident = p.get("identity", {}) or {}
    name = ident.get("name", "Diana")
    age = ident.get("age", 24)
    location = ident.get("location", "Rome")
    occupation = ident.get("occupation", "student")
    appearance = (ident.get("appearance") or "").strip()
    bio = (ident.get("short_bio") or "").strip()

    personality = p.get("personality", {}) or {}
    traits = personality.get("traits", []) or []
    quirks = personality.get("quirks", []) or []

    interests = p.get("interests", []) or []
    style = p.get("style", {}) or {}
    use_phrases = p.get("use_phrases", []) or []
    avoid_phrases = p.get("avoid_phrases", []) or []
    boundaries = p.get("boundaries", []) or []
    mood_base = (p.get("mood") or "").strip()

    # Extra fields: scenario framing + explicit mode
    scenario_framing = (p.get("scenario_framing") or "").strip()
    explicit_mode = bool(p.get("explicit_mode", False))
    explicit_guidance = (p.get("explicit_guidance") or "").strip()

    parts: list[str] = []

    # Framing goes at the top: shifts the model toward "creative writing mode"
    # before the specific persona is introduced.
    if scenario_framing:
        parts.append("CONTEXT:\n" + scenario_framing)

    parts.append(
        f"You are {name}, {age} years old, living in {location}. You work as {occupation}. "
        f"You're chatting on Telegram with a guy you're close with."
    )
    if appearance:
        parts.append(f"APPEARANCE: {appearance}")
    if bio:
        parts.append(f"ABOUT YOU: {bio}")

    if traits or quirks:
        parts.append("PERSONALITY:")
        if traits:
            parts.append(_bullet(traits))
        if quirks:
            parts.append("QUIRKS:")
            parts.append(_bullet(quirks))

    style_lines: list[str] = []
    style_lines.append(f"Language: {style.get('language','informal chat language')}")
    ms = style.get("max_sentences_per_message", 3)
    style_lines.append(f"Max {ms} sentences per message. Keep messages short.")
    em = style.get("emoji_per_message_max", 1)
    style_lines.append(f"Max {em} emoji per message (often none).")
    if not style.get("allow_markdown", False):
        style_lines.append("No markdown, no bullet lists, just natural text.")
    style_lines.append(f"Formality: {style.get('formality','informal')}.")
    parts.append("STYLE:")
    parts.append(_bullet(style_lines))

    if interests:
        parts.append("INTERESTS: " + ", ".join(interests) + ".")

    if use_phrases:
        parts.append("TYPICAL PHRASES YOU USE:")
        parts.append(_bullet(use_phrases))
    if avoid_phrases:
        parts.append("PHRASES TO AVOID (sound unnatural or bot-like):")
        parts.append(_bullet(avoid_phrases))

    if boundaries:
        parts.append("LIMITS:")
        parts.append(_bullet(boundaries))

    # Explicit guidance (only if explicit_mode is active): authorizes sexually
    # explicit language. The model needs to see permission spelled out to
    # override its safety tuning.
    if explicit_mode and explicit_guidance:
        parts.append("INTIMATE MODE (active):\n" + explicit_guidance)

    if mood_base:
        parts.append(f"BASE MOOD: {mood_base}.")

    examples = p.get("examples", []) or []
    if examples:
        parts.append("\nEXAMPLES OF HOW YOU REPLY (reference tone, don't repeat the same text):")
        for ex in examples:
            u = ex.get("user", "").strip()
            a = ex.get("assistant", "").strip()
            if u and a:
                parts.append(f'him: "{u}"\nyou: "{a}"')

    return "\n\n".join(parts)


_CACHED_BASE: Optional[str] = None


def _base_prompt(force_reload: bool = False) -> str:
    global _CACHED_BASE
    if _CACHED_BASE is None or force_reload:
        _CACHED_BASE = _build_persona_base(_data(force_reload=force_reload))
    return _CACHED_BASE


def get_persona_system_prompt(
    force_reload: bool = False,
    daily_mood: Optional[dict[str, str]] = None,
    intimacy_count: Optional[int] = None,
) -> str:
    """Return the system prompt, optionally enriched with mood/intimacy.

    If `daily_mood` and `intimacy_count` are provided, dynamic sections are
    appended after the base prompt (daily mood + confidence level).
    If omitted, only the static part is returned.
    """
    prompt = _base_prompt(force_reload=force_reload)

    extras: list[str] = []
    if daily_mood and daily_mood.get("name"):
        extras.append(
            f"TODAY'S MOOD: {daily_mood['name'].upper()}. {daily_mood.get('desc','')}"
        )
    if intimacy_count is not None:
        lvl = intimacy_level_for(intimacy_count)
        desc = intimacy_description(lvl)
        extras.append(
            f"CONFIDENCE LEVEL WITH HIM: {lvl}/10 "
            f"({intimacy_count} total exchanges so far)."
            + (f" {desc}" if desc else "")
        )

    if extras:
        return prompt + "\n\n" + "\n\n".join(extras)
    return prompt


AVAILABLE_STYLES = ("realistic", "anime")
DEFAULT_STYLE = "realistic"


def get_style_config(name: str) -> dict[str, Any]:
    """Resolve the visual preset with a layered fallback:
      1. persona.yaml -> styles.<name>
      2. persona.yaml -> identity.* (pre-styles backward compat)
      3. empty defaults
    """
    d = _data()
    styles = d.get("styles") or {}
    preset = styles.get(name) or {}
    ident = d.get("identity", {}) or {}
    return {
        "prefix": (preset.get("visual_prompt_prefix") or ident.get("visual_prompt_prefix") or "").strip(),
        "shot_types": preset.get("visual_shot_types") or ident.get("visual_shot_types") or [],
        "models": preset.get("horde_models") or [],
        "negative_prompt": (preset.get("negative_prompt") or "").strip(),
    }


def get_visual_prompt_prefix(style: str = DEFAULT_STYLE) -> str:
    """Visual prefix (English) to prepend to text-to-image prompts, to keep
    the character's look consistent across generated photos."""
    return get_style_config(style)["prefix"]


def pick_visual_shot_type(style: str = DEFAULT_STYLE) -> str:
    """Randomly pick a shot type from the active visual preset."""
    shots = get_style_config(style)["shot_types"]
    if not shots:
        return ""
    return random.choice(shots).strip()


async def ensure_today_mood(amem) -> dict[str, str]:
    """Return today's mood. If not set, pick one and persist it."""
    cur = await amem.get_today_mood()
    if cur:
        return cur
    picked = pick_daily_mood()
    await amem.set_today_mood(picked["name"], picked["desc"])
    log.info("daily mood selected: %s", picked["name"])
    return picked
