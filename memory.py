"""Long-term memory: facts the assistant saves between conversations."""

from datetime import date

import storage

FILE = "memory.json"


def load_memories() -> list:
    return storage.load(FILE, [])


def remember(fact: str) -> dict:
    """Save a fact about the user or their life to long-term memory.

    Args:
        fact: The fact to remember, phrased in third person,
            e.g. "The user's name is Saurabh" or "Exam on Friday 5 Sept".
    """
    memories = load_memories()
    entry = f"{fact} (saved {date.today().isoformat()})"
    memories.append(entry)
    storage.save(FILE, memories)
    return {"saved": entry, "total_memories": len(memories)}


def forget(text_to_forget: str) -> dict:
    """Delete saved memories that contain the given text.

    Args:
        text_to_forget: A word or phrase; every memory containing it is deleted.
    """
    memories = load_memories()
    keep = [m for m in memories if text_to_forget.lower() not in m.lower()]
    removed = len(memories) - len(keep)
    storage.save(FILE, keep)
    return {"deleted": removed, "remaining": len(keep)}
