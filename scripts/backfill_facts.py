"""Retroactively process user messages in the DB to extract facts the bot
should have remembered. Useful the first time after enabling automatic
fact extraction on a DB that already has history.

Usage (stop the bot first to avoid Ollama contention):
    sudo systemctl stop diana-bot
    source venv/bin/activate
    python -m scripts.backfill_facts
    sudo systemctl start diana-bot
"""
import asyncio
import functools
import sys
import time

# Unbuffered output is essential to see progress.
print = functools.partial(print, flush=True)

from src.memory import Memory
from src.agent import extract_facts
from src.config import CONFIG


async def main() -> int:
    mem = Memory(CONFIG.db_path, CONFIG.allowed_chat_id)
    existing = set(mem.list_facts(limit=500))
    print(f"existing facts: {len(existing)}")

    with mem._conn() as c:
        rows = c.execute(
            "SELECT id, content FROM messages WHERE chat_id=? AND role='user' ORDER BY id ASC",
            (mem.chat_id,),
        ).fetchall()
    user_msgs = [(r["id"], r["content"]) for r in rows]
    print(f"user messages in DB: {len(user_msgs)}")
    to_process = [(i, c) for i, c in user_msgs if len(c.strip()) >= 15]
    print(f"to process (len>=15): {len(to_process)}\n")

    added = 0
    t0 = time.time()
    for n, (mid, content) in enumerate(to_process, 1):
        preview = content.replace("\n", " ")[:90]
        print(f"[{n}/{len(to_process)}] #{mid}: {preview}")
        try:
            facts = await extract_facts(content)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        if not facts:
            print("  -> no facts")
            continue
        for f in facts:
            if f in existing:
                print(f"  = {f}  (already present)")
                continue
            mem.save_fact(f)
            existing.add(f)
            added += 1
            print(f"  + {f}")

    dt = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"elapsed: {dt:.1f}s")
    print(f"new facts added: {added}")
    print(f"total facts now: {len(existing)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
