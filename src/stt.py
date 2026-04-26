"""Speech-to-text via faster-whisper (local, CPU-only).

Exposes `transcribe_voice(ogg_bytes)` which returns the text recognised
from a Telegram voice message (OGG Opus format). Pipeline:
  OGG Opus -> ffmpeg -> WAV PCM 16kHz mono -> faster-whisper -> text

The model is loaded lazily on first use (downloaded automatically from
HuggingFace, cached under ~/.cache/huggingface/hub/).
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from src.config import CONFIG


log = logging.getLogger("diana-bot.stt")

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    """Load the whisper model once (singleton)."""
    global _model
    if _model is None:
        log.info("loading faster-whisper model: %s (CPU int8)", CONFIG.stt_model)
        # int8: leaner in RAM and faster on CPU, with negligible quality loss
        # for short clips.
        _model = WhisperModel(CONFIG.stt_model, device="cpu", compute_type="int8")
        log.info("faster-whisper ready")
    return _model


def _ogg_to_wav(ogg_bytes: bytes) -> bytes:
    """Convert OGG Opus -> WAV PCM 16kHz mono (format Whisper expects)."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "ogg", "-i", "pipe:0",
            "-ar", "16000", "-ac", "1",
            "-f", "wav", "pipe:1",
        ],
        input=ogg_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'ignore')[:200]}")
    return proc.stdout


def _transcribe_blocking(wav_bytes: bytes) -> str:
    """Transcribe an in-memory WAV. Blocking — call via asyncio.to_thread."""
    model = _get_model()
    # faster-whisper accepts path or file-like; for in-memory WAV we write
    # a tempfile (most robust path: the decoder needs seekable input).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
        tf.write(wav_bytes)
        tf.flush()
        lang = CONFIG.stt_language or None
        segments, info = model.transcribe(
            tf.name,
            language=lang,
            beam_size=1,                  # greedy: ~2x faster on Pi
            vad_filter=True,              # trim silences/breaths
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    log.info("STT ok: lang=%s prob=%.2f duration=%.1fs -> %d chars",
             info.language, info.language_probability, info.duration, len(text))
    return text


async def transcribe_voice(ogg_bytes: bytes) -> str:
    """Receive OGG Opus (Telegram voice), return transcribed text.
    Thread-offloaded so we don't block the event loop (whisper is CPU-bound)."""
    if not ogg_bytes:
        raise ValueError("empty audio")
    wav_bytes = await asyncio.to_thread(_ogg_to_wav, ogg_bytes)
    text = await asyncio.to_thread(_transcribe_blocking, wav_bytes)
    return text
