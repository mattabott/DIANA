"""Centralized configuration loaded from .env."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    telegram_token: str
    allowed_chat_id: int
    ollama_host: str
    ollama_model: str
    ollama_keep_alive: str
    db_path: Path
    persona_file: Path
    log_level: str
    autonomous_enabled: bool
    autonomous_min_hour: int
    autonomous_max_hour: int
    autonomous_max_per_day: int
    autonomous_check_interval_min: int
    autonomous_weekend: bool
    horde_api_key: str
    horde_models: list
    horde_nsfw: bool
    horde_seed: str
    horde_use_ref: bool
    pic_max_per_day: int
    pic_cooldown_min: int
    pic_autonomous_prob: int
    voice_enabled: bool
    voice_probability: int
    voice_max_chars: int
    tts_voice_model: str
    stt_enabled: bool
    stt_model: str
    stt_language: str


CONFIG = Config(
    telegram_token=_require("TELEGRAM_BOT_TOKEN"),
    allowed_chat_id=int(_require("ALLOWED_CHAT_ID")),
    ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
    ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "24h"),
    db_path=ROOT / os.getenv("DB_PATH", "./data/chatbot.db"),
    persona_file=ROOT / os.getenv("PERSONA_FILE", "./config/persona.yaml"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    autonomous_enabled=_bool("AUTONOMOUS_ENABLED", True),
    autonomous_min_hour=_int("AUTONOMOUS_MIN_HOUR", 9),
    autonomous_max_hour=_int("AUTONOMOUS_MAX_HOUR", 23),
    autonomous_max_per_day=_int("AUTONOMOUS_MAX_PER_DAY", 3),
    autonomous_check_interval_min=_int("AUTONOMOUS_CHECK_INTERVAL_MIN", 30),
    autonomous_weekend=_bool("AUTONOMOUS_WEEKEND", False),
    horde_api_key=os.getenv("HORDE_API_KEY", "0000000000"),
    horde_models=[m.strip() for m in os.getenv("HORDE_MODELS", "").split(",") if m.strip()],
    horde_nsfw=_bool("HORDE_NSFW", True),
    horde_seed=os.getenv("HORDE_SEED", "").strip(),
    horde_use_ref=_bool("HORDE_USE_REF", False),
    pic_max_per_day=_int("PIC_MAX_PER_DAY", 10),
    pic_cooldown_min=_int("PIC_COOLDOWN_MIN", 5),
    pic_autonomous_prob=_int("PIC_AUTONOMOUS_PROB", 30),
    voice_enabled=_bool("VOICE_ENABLED", False),
    voice_probability=_int("VOICE_PROBABILITY", 40),
    voice_max_chars=_int("VOICE_MAX_CHARS", 400),
    tts_voice_model=os.getenv("TTS_VOICE_MODEL", ""),
    stt_enabled=_bool("STT_ENABLED", False),
    stt_model=os.getenv("STT_MODEL", "small"),
    stt_language=os.getenv("STT_LANGUAGE", ""),
)
