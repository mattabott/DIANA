# Diana

Personal **single-user** Telegram bot powered by a local LLM (Ollama), with configurable persona, persistent memory, autonomous messages and photo generation via Stable Horde. Built for private roleplay / companion use between consenting adults.

> **Adult content warning** — the system optionally supports explicit roleplay and NSFW image generation through Stable Horde. It is meant for **personal, local, adult use only**. It is not a product intended for public or multi-user deployment.

---

## Features

- **Local LLM via Ollama** (recommended model: `qwen3.5:4b`, multimodal)
- **LangGraph ReAct agent** with tool calling
- **Fully configurable persona** in `config/persona.yaml` (identity, style, mood, examples, rules)
- **Layered SQLite memory**: raw messages + durable facts + dated events + daily mood + interaction stats
- **Automatic fact extraction** in background after each user message (LLM extracts personal info)
- **Multimodality**: the bot receives and "sees" photos you send (contextual caption)
- **Autonomous messages** (APScheduler): the bot writes you spontaneously during the day
- **Daily mood**: a mood is randomly picked every morning and influences the tone
- **Intimacy level 1-10** that grows with the number of exchanges, modulating tone familiarity
- **Photo generation** via Stable Horde (free with anonymous API):
  - `/pic <english prompt>` — explicit prompt
  - `/selfie <hint>` — LLM generates scene + caption
  - **Automatic detection** of photo requests in chat text ("send me a picture...")
  - **Context-aware**: if the bot just mentioned a photo and you insist ("go on, send it"), it triggers
  - **Autonomous**: N% of autonomous ticks become spontaneous photos
- **Explicit mode** on/off: a flag in the YAML enables/disables NSFW language and prompts
- **systemd deployment** ready (`deploy/diana-bot.service`)

## Architecture

```
Telegram ──► Bot (chat_id filter) ──► LangGraph ReAct agent
                │                         │
                ├── on_text                ├── ChatOllama (text)
                ├── on_photo (vision)      ├── tools: datetime, remember_this
                ├── /pic /selfie /...      └── DB memory
                └── /fact /event /memory   
                                           
                  Scheduler (APScheduler) ──► autonomous msg / photo
                                           
  SQLite: messages, facts, events, daily_mood, stats, pic_log, autonomous_log
  
  Horde client ──► Stable Horde API ──► photo (img2img / txt2img)
```

---

## Requirements

### Hardware

- **Raspberry Pi 5 (8 GB RAM)** or equivalent ARM64 / x86_64 Linux machine
- Storage: at least 10 GB free (model ~3.4 GB + venv + data)
- Network access for Telegram + Stable Horde

On more powerful hardware (laptop with GPU, x86 server) performance scales linearly.

### Software

- Linux (tested on Raspberry Pi OS bookworm)
- Python 3.11+
- [Ollama](https://ollama.com/) ≥ 0.20
- systemd (for persistent deployment)
- git, curl

---

## Step-by-step setup

### 1. Ollama and model

```bash
# Install Ollama (Linux / ARM64)
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version

# Pull the recommended model (~3.4 GB)
ollama pull qwen3.5:4b
```

**Model alternatives** (edit `OLLAMA_MODEL` in `.env`):
- `qwen3.5:2b` — faster, lower quality
- `qwen3-vl:4b` — older, still multimodal

### 2. Clone and venv

```bash
git clone git@github.com:mattabott/DIANA.git diana
cd diana

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### 3. Telegram bot

1. On Telegram open **@BotFather** → `/newbot` → follow the prompts
2. Copy the **bot token**
3. Open **@userinfobot** → `/start` → copy your numeric **user id**

### 4. Configuration

```bash
cp .env.example .env
cp config/persona.yaml.example config/persona.yaml
```

Edit `.env`:
- `TELEGRAM_BOT_TOKEN=<the bot token>`
- `ALLOWED_CHAT_ID=<your user id>`
- other fields have sensible defaults

Edit `config/persona.yaml`:
- `identity.name`, `location`, `occupation`, `appearance`, `short_bio`
- `visual_prompt_prefix` (English) if you want a specific look for generated photos
- `explicit_mode: true|false` to enable/disable adult content
- `personality.traits`, `use_phrases`, `avoid_phrases`, `examples`
- `daily_moods`, `intimacy_levels` → editable

### 5. Stable Horde (photos, optional)

The bot generates photos through [Stable Horde](https://stablehorde.net) using the anonymous API key `0000000000` → works immediately but with low priority (photos in 2-10 min).

To speed things up:
1. Register an account at https://stablehorde.net/register
2. Copy your personal API key
3. Paste it in `.env` under `HORDE_API_KEY`

### 6. Quick test

```bash
source venv/bin/activate
python -m src.bot
```

Open Telegram → write to your bot → watch logs in the terminal. `Ctrl+C` to stop.

### 7. Deploy as systemd service

```bash
# Run from the project root — install.sh substitutes <USER> and <PROJECT_DIR>
# placeholders in the service file automatically.
sudo ./deploy/install.sh
```

The script:
1. Copies `deploy/diana-bot.service` to `/etc/systemd/system/`
2. Runs `systemctl daemon-reload`
3. Runs `enable --now` → the bot starts immediately **and** at system boot

**Useful commands:**
```bash
sudo systemctl status diana-bot       # status
sudo journalctl -u diana-bot -f       # live logs
sudo systemctl restart diana-bot      # after persona.yaml or code changes
sudo systemctl stop diana-bot         # stop
./deploy/uninstall.sh                 # remove completely
```

---

## Available commands (in chat)

| Command | What it does |
|---------|--------------|
| `/start` | Initial greeting |
| `/memory` | Show everything the bot knows about you (facts, events, stats) |
| `/fact <text>` | Save a durable fact (always injected into the prompt) |
| `/event <text>` | Save a recent event (injected into the prompt for 7 days) |
| `/pic <english prompt>` | Generate a photo with explicit prompt |
| `/selfie [hint]` | Generate a photo with scene decided by the LLM |
| `/refs` | List currently loaded reference photos |
| `/setref` | Set a reference (send a photo with caption `/setref`) |
| `/clearref` | Remove all references |
| `/ping` | Force an immediate autonomous message (debug) |

Additionally: the bot **automatically detects** photo requests in chat text (e.g., *"send me a photo"*, *"show me"*), even as conversational continuations.

---

## Advanced configuration

### Change the persona
All text changes about personality, name, city, style go in `config/persona.yaml`. After each change:
```bash
sudo systemctl restart diana-bot
```

### Enable/disable explicit mode
`config/persona.yaml` → `explicit_mode: true|false`. The `scenario_framing` and `explicit_guidance` are injected only when `true`. Combine with explicit `examples` for better results.

### Photo tuning
In `.env`:
- `HORDE_MODELS`: ordered list of preferred Horde models
- `HORDE_SEED`: fixed number → more consistent subject across photos
- `HORDE_NSFW`: true/false for the NSFW flag in Horde requests

In `persona.yaml`:
- `identity.visual_prompt_prefix`: English description of the subject (**always adult**: use `adult`, `26 year old`, etc. **NEVER** `young/teen/youthful` — even as negatives they trigger Horde CSAM filter)
- `identity.visual_shot_types`: random composition patterns (selfie / mirror / etc.)

### Memory
- **Facts** (durable): inserted via `/fact` or auto-extracted by the LLM after each user message
- **Events** (recent): inserted via `/event`, drop out of the prompt after 7 days
- **Mood**: randomly picked at the start of the day, persists until midnight
- **Intimacy**: `1 + interaction_count // 20` (max 10)

### Autonomous messages
`.env`:
- `AUTONOMOUS_ENABLED`: master on/off
- `AUTONOMOUS_MIN_HOUR`/`MAX_HOUR`: allowed time window
- `AUTONOMOUS_MAX_PER_DAY`: quota
- `AUTONOMOUS_CHECK_INTERVAL_MIN`: check frequency
- `AUTONOMOUS_WEEKEND`: true/false to run (or not) during Saturday+Sunday
- `PIC_AUTONOMOUS_PROB`: probability (%) that an autonomous tick becomes a photo instead of text

---

## Troubleshooting

| Issue | Cause / fix |
|-------|-------------|
| **Bot does not respond** | Check `sudo journalctl -u diana-bot -f`. If Ollama is down: `sudo systemctl status ollama` |
| **Extremely slow replies (>3 min)** | First model load can take 1-2 min. `OLLAMA_KEEP_ALIVE=24h` avoids re-load |
| **Timeout after 600s** | Context too large. Reduce `HISTORY_WINDOW` in `src/bot.py` or reduce `num_ctx` |
| **`CorruptPrompt` from Horde** | Your prompt contains words that trigger filters (especially CSAM). Avoid `teen`, `young`, `schoolgirl` even in negative prompts. IP is timed out for ~2 min afterwards |
| **Photo arrives as black banner / censored** | Horde CSAM post-filter. The bot **detects** files < 25 KB and replies `"it came out wrong, try again"`. Reinforce `adult/mature` in `visual_prompt_prefix` |
| **`KudosUpfront`** | Requests > 576px or > 50 steps require kudos. The 384x576 default is fine. Do not increase to 768 without a registered account and kudos |
| **Very long Horde queue (>10 min)** | Not much use waiting with anon key. Register at stablehorde.net for initial kudos |
| **Safe scenes when asking for NSFW** | The LLM refused. The keyword fallback should handle `tits/nude/topless/lingerie/shower/bed`; add more keywords in `src/pic_prompt.py:_EXPLICIT_KEYWORD_MAP` if needed |

---

## Security / privacy

- `.env` contains Telegram token and chat_id → **never commit** (it's in `.gitignore`)
- `data/chatbot.db` contains conversation history → **never commit**
- `data/refs/` contains reference photos → **never commit**
- The bot hard-filters on `ALLOWED_CHAT_ID`: any other chat is ignored with `UNAUTHORIZED` in logs
- All processing (LLM, DB, persona) is **local** on your machine. Only Telegram API and Stable Horde see outbound traffic

---

## Project structure

```
.
├── .env.example               # copy to .env and customize
├── .gitignore
├── config/
│   └── persona.yaml.example   # copy to persona.yaml and customize
├── deploy/
│   ├── diana-bot.service      # systemd unit
│   ├── install.sh             # deployment (requires sudo)
│   └── uninstall.sh
├── scripts/
│   ├── backfill_facts.py      # retroactive fact extraction from DB history
│   ├── bench_models.sh        # Ollama model benchmark
│   └── test_ollama.py         # LLM smoke test (text + image)
├── src/
│   ├── __init__.py
│   ├── agent.py               # ReAct agent + vision LLM + fact extraction
│   ├── bot.py                 # entrypoint, Telegram handlers
│   ├── config.py              # .env loader
│   ├── horde.py               # Stable Horde client
│   ├── memory.py              # SQLite (Memory + AsyncMemory)
│   ├── persona.py             # YAML loader → system prompt
│   ├── pic_flow.py            # photo orchestration (intent, flow, rate limit)
│   ├── pic_prompt.py          # LLM scene + caption generator
│   ├── scheduler.py           # autonomous messages (text + photo)
│   └── tools.py               # ReAct agent tools
└── requirements.txt
```

---

## Disclaimer

Experimental software for personal use. Not tested for multi-user scalability or public deployment. The author is not responsible for generated content: LLM and Stable Diffusion models can produce unexpected output. Use responsibly and in compliance with local laws and the Terms of Service of Telegram, Ollama, and Stable Horde.
