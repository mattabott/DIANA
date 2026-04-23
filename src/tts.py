"""Text-to-speech via Piper (local, CPU-only).

Exposes `synthesize_opus(text)` which returns OGG Opus bytes ready for
Telegram's `bot.send_voice()` (the format required to display the audio
as a native "voice message" with waveform, not a generic audio file).

Piper produces WAV PCM 22050Hz mono; ffmpeg re-encodes to OGG Opus at
48kHz 24kbps (Telegram voice defaults).
"""
from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import wave
from pathlib import Path
from typing import Optional

from piper import PiperVoice

from src.config import CONFIG


log = logging.getLogger("diana-bot.tts")


_VOICE_DIR = CONFIG.db_path.parent / "voices"
_voice: Optional[PiperVoice] = None


def _get_voice() -> PiperVoice:
    """Load the Piper voice once (singleton). Path resolved from .env."""
    global _voice
    if _voice is None:
        model_path = _VOICE_DIR / CONFIG.tts_voice_model
        if not model_path.exists():
            raise RuntimeError(
                f"Piper voice model not found at {model_path}. "
                f"Download the .onnx + .onnx.json from https://huggingface.co/rhasspy/piper-voices"
            )
        _voice = PiperVoice.load(str(model_path))
        log.info("Piper voice loaded: %s", CONFIG.tts_voice_model)
    return _voice


def _synthesize_wav_blocking(text: str) -> bytes:
    """Synthesize text into WAV bytes (blocking). Run via asyncio.to_thread."""
    v = _get_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        v.synthesize_wav(text, wf)
    return buf.getvalue()


def _wav_to_opus_blocking(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes -> OGG Opus bytes via an ffmpeg subprocess.
    Telegram voice format: 48kHz, mono, ~24-32kbps opus.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "wav", "-i", "pipe:0",
            "-c:a", "libopus", "-b:a", "24k", "-ar", "48000", "-ac", "1",
            "-f", "ogg", "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'ignore')[:200]}")
    return proc.stdout


async def synthesize_opus(text: str) -> bytes:
    """Synthesize text -> OGG Opus bytes (for Telegram send_voice).
    Thread-offloaded to avoid blocking the event loop.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty text")
    wav_bytes = await asyncio.to_thread(_synthesize_wav_blocking, text)
    opus_bytes = await asyncio.to_thread(_wav_to_opus_blocking, wav_bytes)
    log.debug("TTS ok: text=%d chars -> wav=%d B -> opus=%d B",
              len(text), len(wav_bytes), len(opus_bytes))
    return opus_bytes
