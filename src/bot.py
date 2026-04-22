"""Telegram bot entrypoint: single-user filter, handlers, scheduler wiring."""
import asyncio
import base64
import io
import logging
import re
from pathlib import Path
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from PIL import Image

from src.config import CONFIG
from src.agent import generate_reply, generate_vision_reply, extract_facts
from src.memory import Memory, AsyncMemory
from src.tools import set_memory
from src.scheduler import maybe_send_autonomous, force_send_autonomous
from src.persona import ensure_today_mood, get_visual_prompt_prefix, pick_visual_shot_type
from src.horde import generate_image, HordeError
from src.pic_flow import is_photo_request, is_photo_continuation, run_pic_flow, can_send_pic


MAX_IMAGE_SIDE = 512  # resize target: quality/speed trade-off on Pi 5

REFS_DIR = CONFIG.db_path.parent / "refs"
REFS_DIR.mkdir(parents=True, exist_ok=True)
MAX_REFS = 5
REF_MAX_SIDE = 768  # references are larger than vision inputs for Horde quality


logging.basicConfig(
    level=getattr(logging, CONFIG.log_level),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("diana-bot")


# How many past messages to include in each turn's context.
HISTORY_WINDOW = 6


def _authorized(update: Update) -> bool:
    chat = update.effective_chat
    if chat is None or chat.id != CONFIG.allowed_chat_id:
        user = update.effective_user
        log.warning(
            "UNAUTHORIZED access attempt: chat_id=%s user_id=%s username=%s",
            getattr(chat, "id", None),
            getattr(user, "id", None),
            getattr(user, "username", None),
        )
        return False
    return True


async def _keepalive_typing(bot, chat_id: int, stop: asyncio.Event) -> None:
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


def _get_memory(context: ContextTypes.DEFAULT_TYPE) -> AsyncMemory:
    return context.application.bot_data["memory"]


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    await update.message.reply_text("hi 👋")


async def on_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Debug command: force an immediate autonomous message (bypasses rate limit)."""
    if not _authorized(update):
        return
    amem = _get_memory(context)
    log.info("/ping forcing autonomous message")
    try:
        await force_send_autonomous(context.bot, amem)
    except Exception:
        log.exception("/ping failed")
        await update.message.reply_text("(ping failed)")


async def on_fact_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/fact <text>: save a durable fact about the user (no in-character reply)."""
    if not _authorized(update):
        return
    text = " ".join(context.args).strip() if context.args else ""
    if not text:
        await update.message.reply_text("usage: /fact <durable fact to remember>")
        return
    amem = _get_memory(context)
    await amem.save_fact(text)
    log.info("manual /fact saved: %s", text[:120])
    await update.message.reply_text(f"✓ fact saved: {text}")


async def on_event_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/event <text>: save a recent event (stays in prompt for ~7 days)."""
    if not _authorized(update):
        return
    text = " ".join(context.args).strip() if context.args else ""
    if not text:
        await update.message.reply_text("usage: /event <recent event to remember>")
        return
    amem = _get_memory(context)
    await amem.save_event(text)
    log.info("manual /event saved: %s", text[:120])
    await update.message.reply_text(f"✓ event saved: {text}")


async def on_selfie_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/selfie <hint> — force a reactive photo with an optional hint (bypasses regex).
    Unlike /pic: the LLM generates the English scene prompt and a caption
    starting from your hint (if any) or just the current mood.
    """
    if not _authorized(update):
        return
    hint = " ".join(context.args).strip() if context.args else ""
    amem = _get_memory(context)
    ok, reason = await can_send_pic(amem)
    if not ok:
        await update.message.reply_text(_photo_refusal_message(reason))
        return
    mood = await ensure_today_mood(amem)
    caption = await run_pic_flow(
        bot=context.bot,
        amem=amem,
        chat_id=update.message.chat_id,
        trigger_type="reactive",
        user_hint=hint or None,
        mood=mood,
    )
    if caption is None:
        await update.message.reply_text("mmh can't manage it right now, try again")


# Clothing / outfit words. If the user prompt is "outfit-only" (no scene,
# pose, or framing), route to the LLM so it can build a proper scene with
# a full-body shot, otherwise SD tends to pick head-only selfies.
_OUTFIT_WORDS = {
    "bathrobe", "towel", "pajamas", "pyjamas", "pjs", "robe", "nightgown",
    "dress", "skirt", "shirt", "blouse", "t-shirt", "tshirt", "top",
    "hoodie", "sweater", "jacket", "coat", "suit", "tracksuit",
    "jeans", "pants", "shorts", "leggings", "tights",
    "lingerie", "bra", "panties", "underwear", "bikini", "swimsuit", "thong",
    "stockings", "pantyhose", "socks",
    "heels", "boots", "shoes", "sneakers",
    "uniform", "costume", "outfit", "nude", "naked", "topless",
}

# SD-style descriptors. If present, the prompt likely is already a proper
# Stable Diffusion prompt, so use it literal.
_SD_STYLE_HINTS = {
    "wearing", "sitting", "standing", "holding", "lying", "looking",
    "leaning", "kneeling", "walking", "posing",
}


def _prompt_needs_llm(prompt: str) -> bool:
    """True if the prompt should be routed to the LLM scene-generator
    instead of sent literal to Horde. Two cases:
      1) Short prompt with an outfit word but no scene/pose descriptor
         (e.g. "sexy bathrobe"): without elaboration, SD picks random
         framing and the outfit often ends up out of frame.
      2) Very short prompt (<=3 words): too minimal for SD to interpret.
    """
    words = re.findall(r"\b[a-zA-Z]+\b", prompt.lower())
    word_set = set(words)
    has_sd_scene = bool(word_set & _SD_STYLE_HINTS)
    has_outfit = bool(word_set & _OUTFIT_WORDS)
    if len(words) <= 6 and has_outfit and not has_sd_scene:
        return True
    if len(words) <= 3:
        return True
    return False


async def on_pic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/pic <prompt> — generate a photo.
    If the prompt is SD-style (wearing/sitting/standing + commas), sends it
    literally to Horde. If it's short/outfit-only (e.g. "sexy bathrobe"),
    routes it through the LLM scene-generator to elaborate scene+pose+framing.
    """
    if not _authorized(update):
        return
    user_prompt = " ".join(context.args).strip() if context.args else ""
    if not user_prompt:
        await update.message.reply_text(
            "usage: /pic <description>\n"
            "- SD-style English: /pic wearing a red dress, in a bookstore\n"
            "- short hint: /pic sexy bathrobe  (LLM will elaborate the scene)"
        )
        return
    amem = _get_memory(context)
    ok, reason = await can_send_pic(amem)
    if not ok:
        await update.message.reply_text(f"⏳ quota/cooldown: {reason}")
        return

    if _prompt_needs_llm(user_prompt):
        log.info("/pic routing to LLM scene generator: %r", user_prompt[:80])
        mood = await ensure_today_mood(amem)
        caption = await run_pic_flow(
            bot=context.bot,
            amem=amem,
            chat_id=update.message.chat_id,
            trigger_type="manual",
            user_hint=user_prompt,
            mood=mood,
            status_message_text=f"⏳ generating (LLM-elaborated)... ({user_prompt[:80]})",
        )
    else:
        caption = await run_pic_flow(
            bot=context.bot,
            amem=amem,
            chat_id=update.message.chat_id,
            trigger_type="manual",
            explicit_prompt=user_prompt,
            status_message_text=f"⏳ generating... ({user_prompt[:80]})",
        )
    if caption is None:
        # error already shown via status message in run_pic_flow
        pass


async def on_refs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/refs: list currently loaded reference photos."""
    if not _authorized(update):
        return
    refs = _list_refs()
    if not refs:
        await update.message.reply_text(
            "no references set. Send a photo with caption /setref to add one."
        )
        return
    lines = [f"📸 {len(refs)}/{MAX_REFS} active references:"]
    for r in refs:
        size_kb = r.stat().st_size / 1024
        lines.append(f"  · {r.name} ({size_kb:.0f} KB)")
    lines.append("\nUse /clearref to remove all. Send a photo with caption /setref to add more.")
    await update.message.reply_text("\n".join(lines))


async def on_clearref_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/clearref: delete all reference photos."""
    if not _authorized(update):
        return
    refs = _list_refs()
    for r in refs:
        r.unlink(missing_ok=True)
    log.info("cleared %d references", len(refs))
    await update.message.reply_text(f"✓ removed {len(refs)} references. /pic reverts to txt2img.")


async def on_memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/memory: show a summary of what the bot knows about you."""
    if not _authorized(update):
        return
    amem = _get_memory(context)
    facts = await amem.list_facts(limit=50)
    events = await amem.recent_events(days=7, limit=50)
    count = await amem.get_interaction_count()
    pics_today = await amem.pic_count_today()
    lines = [f"📊 total exchanges: {count}"]
    lines.append(f"📸 photos today: {pics_today}/{CONFIG.pic_max_per_day}")
    lines.append(f"\n🔹 facts ({len(facts)}):")
    for f in facts:
        lines.append(f"  · {f}")
    lines.append(f"\n🔸 events in last 7 days ({len(events)}):")
    for e in events:
        lines.append(f"  · [{e['created_at'][:16]}] {e['text']}")
    await update.message.reply_text("\n".join(lines) if lines else "memory is empty")


def _photo_refusal_message(reason: str) -> str:
    """In-character message when the bot can't send a photo (quota / cooldown)."""
    if "quota" in reason:
        return "mmh not tonight, enough photos for today 😏"
    if "cooldown" in reason:
        return "ahah hold on a sec, I'm not doing a photoshoot"
    return "mmh not in the mood right now"


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    msg = update.message
    user_text = msg.text or ""
    log.info("TEXT from chat_id=%s: %s", msg.chat_id, user_text)

    amem = _get_memory(context)

    # Photo intent detection: two paths
    #   a) direct regex on the message ("send me a photo...")
    #   b) hybrid: bot's last message mentioned a photo + user insists
    is_request = is_photo_request(user_text)
    is_continuation = False
    if not is_request:
        last_bot = await amem.last_assistant_text()
        is_continuation = is_photo_continuation(user_text, last_bot)
        if is_continuation:
            log.info("photo continuation detected (last bot said: %r)", last_bot[:80])

    if is_request or is_continuation:
        log.info("photo request detected in: %r", user_text[:80])
        # Save user message to history before processing.
        await amem.save_message("user", user_text)

        ok, reason = await can_send_pic(amem)
        mood = await ensure_today_mood(amem)
        if not ok:
            # In-character refusal (quota/cooldown reached).
            refusal = _photo_refusal_message(reason)
            await msg.reply_text(refusal)
            await amem.save_message("assistant", refusal)
            await amem.increment_interaction()
            return

        caption = await run_pic_flow(
            bot=context.bot,
            amem=amem,
            chat_id=msg.chat_id,
            trigger_type="reactive",
            user_hint=user_text,
            mood=mood,
        )
        if caption is None:
            await msg.reply_text("mmh sorry, can't take it right now")
            await amem.save_message("assistant", "mmh sorry, can't take it right now")
        await amem.increment_interaction()
        return
    history, facts, summary, mood, count, events = await asyncio.gather(
        amem.recent_messages(limit=HISTORY_WINDOW),
        amem.list_facts(limit=50),
        amem.latest_summary(),
        ensure_today_mood(amem),
        amem.get_interaction_count(),
        amem.recent_events(days=7, limit=20),
    )
    log.debug(
        "Context: history=%d facts=%d events=%d mood=%s intimacy_count=%d",
        len(history), len(facts), len(events), mood.get("name"), count,
    )

    stop = asyncio.Event()
    typing_task = asyncio.create_task(_keepalive_typing(context.bot, msg.chat_id, stop))
    try:
        reply = await asyncio.wait_for(
            generate_reply(
                user_text=user_text,
                history=history,
                facts=facts,
                summary=summary,
                daily_mood=mood,
                intimacy_count=count,
                events=events,
            ),
            timeout=600.0,
        )
    except asyncio.TimeoutError:
        log.error("Agent timeout after 600s")
        reply = "mmh sorry, I got distracted... what were you saying?"
    except Exception:
        log.exception("Agent error")
        reply = "mmh I'm a bit out of it, try again"
    finally:
        stop.set()
        await typing_task

    await msg.reply_text(reply)

    # Save the exchange AFTER sending (if send fails, better not to memorize).
    await amem.save_message("user", user_text)
    await amem.save_message("assistant", reply)
    await amem.increment_interaction()
    # Fire-and-forget fact extraction in background: does not block chat.
    asyncio.create_task(_background_extract_facts(user_text, amem))


async def _background_extract_facts(user_text: str, amem: AsyncMemory) -> None:
    """Secondary LLM pass to save personal facts. Runs after the reply."""
    try:
        existing = set(await amem.list_facts(limit=500))
        facts = await extract_facts(user_text)
        for f in facts:
            if f in existing:
                continue
            await amem.save_fact(f)
            log.info("fact saved: %s", f[:120])
    except Exception:
        log.exception("background fact extraction failed")


def _resize_and_b64(raw_bytes: bytes, max_side: int = MAX_IMAGE_SIDE) -> str:
    """Resize (if needed) and return base64 JPEG."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _save_reference_photo(raw_bytes: bytes) -> Path:
    """Save a photo to data/refs/ as img2img reference.
    Rotates: if MAX_REFS files already exist, overwrites the oldest.
    Resizes to REF_MAX_SIDE to keep quality without wasting bandwidth.
    """
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size
    scale = min(REF_MAX_SIDE / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    existing = sorted(REFS_DIR.glob("ref_*.jpg"), key=lambda p: p.stat().st_mtime)
    if len(existing) >= MAX_REFS:
        # Overwrite the oldest.
        target = existing[0]
    else:
        # New slot.
        used = {int(p.stem.split("_")[1]) for p in existing if p.stem.split("_")[1].isdigit()}
        idx = next(i for i in range(1, MAX_REFS + 1) if i not in used)
        target = REFS_DIR / f"ref_{idx}.jpg"

    img.save(target, format="JPEG", quality=90)
    return target


def _list_refs() -> list[Path]:
    return sorted(REFS_DIR.glob("ref_*.jpg"))


def _pick_reference_b64() -> str | None:
    """Pick a random reference and return its JPEG as base64, or None if none."""
    refs = _list_refs()
    if not refs:
        return None
    import random
    chosen = random.choice(refs)
    return base64.b64encode(chosen.read_bytes()).decode()


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    msg = update.message
    caption = (msg.caption or "").strip()
    log.info("PHOTO from chat_id=%s caption=%r n_sizes=%d", msg.chat_id, caption, len(msg.photo))

    # Download the highest-resolution photo (last element).
    tg_file = await msg.photo[-1].get_file()
    raw = bytes(await tg_file.download_as_bytearray())
    log.debug("Downloaded photo: %d bytes", len(raw))

    # If the caption starts with /setref, treat it as an img2img reference
    # and do NOT feed it to the bot for comment.
    if caption.lower().startswith("/setref"):
        target = await asyncio.to_thread(_save_reference_photo, raw)
        refs = _list_refs()
        log.info("reference saved to %s (total %d)", target.name, len(refs))
        await msg.reply_text(f"✓ reference saved ({target.name}). Total: {len(refs)}/{MAX_REFS}.")
        return

    image_b64 = await asyncio.to_thread(_resize_and_b64, raw)
    log.debug("Resized photo b64 len: %d", len(image_b64))

    amem = _get_memory(context)
    facts, mood, count, events = await asyncio.gather(
        amem.list_facts(limit=50),
        ensure_today_mood(amem),
        amem.get_interaction_count(),
        amem.recent_events(days=7, limit=20),
    )

    stop = asyncio.Event()
    typing_task = asyncio.create_task(_keepalive_typing(context.bot, msg.chat_id, stop))
    try:
        reply = await asyncio.wait_for(
            generate_vision_reply(
                caption=caption, image_b64=image_b64, facts=facts,
                daily_mood=mood, intimacy_count=count, events=events,
            ),
            timeout=420.0,  # vision on CPU-only Pi 5 is slow: up to 7 min
        )
    except asyncio.TimeoutError:
        log.error("Vision timeout after 420s")
        reply = "hold on, looking at it... gimme a sec"
    except Exception:
        log.exception("Vision error on photo")
        reply = "mmh can't see the image clearly, try again"
    finally:
        stop.set()
        await typing_task

    await msg.reply_text(reply)
    # Save to DB as text ("[photo]" + optional caption).
    user_log = f"[photo]{' caption: ' + caption if caption else ''}"
    await amem.save_message("user", user_log)
    await amem.save_message("assistant", reply)
    await amem.increment_interaction()


async def on_any(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    log.debug("OTHER update: %s", update.to_dict())


async def _autonomous_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    """JobQueue callback: called periodically to possibly send autonomous content."""
    amem: AsyncMemory = context.application.bot_data["memory"]
    try:
        await maybe_send_autonomous(context.bot, amem)
    except Exception:
        log.exception("autonomous tick failed")


def build_app() -> Application:
    # Memory init (single instance for the authorized chat).
    mem_sync = Memory(db_path=CONFIG.db_path, chat_id=CONFIG.allowed_chat_id)
    set_memory(mem_sync)
    amem = AsyncMemory(mem_sync)

    app = Application.builder().token(CONFIG.telegram_token).build()
    app.bot_data["memory"] = amem

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("ping", on_ping))
    app.add_handler(CommandHandler("fact", on_fact_cmd))
    app.add_handler(CommandHandler("event", on_event_cmd))
    app.add_handler(CommandHandler("memory", on_memory_cmd))
    app.add_handler(CommandHandler("pic", on_pic_cmd))
    app.add_handler(CommandHandler("selfie", on_selfie_cmd))
    app.add_handler(CommandHandler("refs", on_refs_cmd))
    app.add_handler(CommandHandler("clearref", on_clearref_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.ALL, on_any))

    # Autonomous job scheduling.
    interval_s = max(60, CONFIG.autonomous_check_interval_min * 60)
    # First tick with 5-min delay after startup (let the bot settle).
    app.job_queue.run_repeating(
        _autonomous_tick,
        interval=interval_s,
        first=300.0,
        name="autonomous_tick",
    )
    log.info("Autonomous scheduled: every %ds (first in 300s)", interval_s)
    return app


def main() -> None:
    log.info("Bot starting. Allowed chat_id=%s model=%s", CONFIG.allowed_chat_id, CONFIG.ollama_model)
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
