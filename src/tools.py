"""Tools available to the ReAct agent.

Tools access the Memory instance through a global singleton set at bot
startup (set_memory()). This avoids passing dynamic state through the
ReAct signatures, which don't handle runtime parameters well.
"""
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool

from src.memory import Memory


_memory: Optional[Memory] = None


def set_memory(m: Memory) -> None:
    global _memory
    _memory = m


def _require_memory() -> Memory:
    if _memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() at startup.")
    return _memory


@tool
def get_current_datetime() -> str:
    """Return the current date and time in a human-readable format.
    Use ONLY if the user asks what day/time it is, or if you need to
    propose a specific time for an appointment.
    """
    now = datetime.now()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    weekday = days[now.weekday()]
    month = months[now.month - 1]
    return f"{weekday} {now.day} {month} {now.year}, {now.hour:02d}:{now.minute:02d}"


@tool
def remember_this(fact: str) -> str:
    """Save an important fact about the user you're chatting with.
    Use when he shares long-term info: name, job, strong preferences
    (favorite music/film/food), important people, dates, places.
    DO NOT save trivia or small talk.
    Write the fact as a short, clear sentence.
    Example: "His name is Matt, lives in Milan, works in AI".
    """
    mem = _require_memory()
    mem.save_fact(fact.strip())
    return f"ok, remembered: {fact.strip()}"


ALL_TOOLS = [get_current_datetime, remember_this]
