"""Actions the cloud brain can't perform itself, so it hands them to a device.

Phone actions ride back in the /chat reply (the phone is the one asking).
PC actions wait in a queue for the laptop agent to pick up.
"""

import time

_phone_queue: list = []
_pc_queue: list = []
_agent_last_seen: float = 0.0

AGENT_TIMEOUT_SECONDS = 20


# ---------- plumbing ----------

def start_request() -> None:
    _phone_queue.clear()


def collect_phone_actions() -> list:
    return list(_phone_queue)


def take_pc_actions() -> list:
    """Called by the laptop agent; empties the queue."""
    global _agent_last_seen
    _agent_last_seen = time.time()
    pending, _pc_queue[:] = list(_pc_queue), []
    return pending


def pc_agent_online() -> bool:
    return (time.time() - _agent_last_seen) < AGENT_TIMEOUT_SECONDS


def _queue_phone(action: dict) -> None:
    """The model sometimes asks for the same thing twice in one turn, which
    would set two alarms or send a message twice. Once is enough — and a
    differing label doesn't make it a different alarm."""

    def identity(item: dict) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in item.items() if k != "label"))

    if any(identity(queued) == identity(action) for queued in _phone_queue):
        return
    _phone_queue.append(action)


def _queue_pc(action: dict, description: str) -> dict:
    if not pc_agent_online():
        return {
            "error": "The PC agent is not running, so the laptop can't be "
            "controlled right now. Tell the user to start Shaurya's agent on "
            "the laptop."
        }
    _pc_queue.append(action)
    return {"sent_to_pc": description}


# ---------- phone tools ----------

def open_app_on_phone(app_name: str) -> dict:
    """Open an app on the user's phone.

    Args:
        app_name: The app to open, e.g. "youtube", "whatsapp", "instagram",
            "chrome", "camera", "settings", or any installed app's name.
    """
    _queue_phone({"action": "open_app", "app": app_name})
    return {"queued_on_phone": f"open {app_name}"}


def open_website_on_phone(url: str) -> dict:
    """Open a website in the phone's browser.

    Args:
        url: The website address, e.g. "youtube.com".
    """
    _queue_phone({"action": "open_url", "url": url})
    return {"queued_on_phone": f"open {url}"}


def set_flashlight(on: bool) -> dict:
    """Turn the phone's flashlight (torch) on or off.

    Args:
        on: True to turn the flashlight on, False to turn it off.
    """
    _queue_phone({"action": "flashlight", "on": bool(on)})
    return {"queued_on_phone": f"flashlight {'on' if on else 'off'}"}


def set_alarm(hour: int, minute: int, label: str = "") -> dict:
    """Set an alarm on the user's phone.

    Args:
        hour: Hour in 24-hour time, so 9am is 9 and 9pm is 21.
        minute: Minute of the hour, 0-59.
        label: Optional short name for the alarm, e.g. "gym".
    """
    _queue_phone(
        {"action": "alarm", "hour": int(hour), "minute": int(minute), "label": label}
    )
    return {"queued_on_phone": f"alarm at {int(hour):02d}:{int(minute):02d}"}


def set_timer(minutes: int, label: str = "") -> dict:
    """Start a countdown timer on the user's phone.

    Args:
        minutes: How many minutes to count down.
        label: Optional short name, e.g. "pasta".
    """
    _queue_phone(
        {"action": "timer", "seconds": int(minutes) * 60, "label": label}
    )
    return {"queued_on_phone": f"{minutes} minute timer"}


def send_whatsapp(contact_name: str, message: str) -> dict:
    """Send a WhatsApp message to one of the user's contacts.

    Args:
        contact_name: The person's name as the user says it, e.g. "Mirage".
            The phone looks the name up in its own contacts, so you never see
            and never need their phone number.
        message: What to say to them.
    """
    _queue_phone(
        {"action": "whatsapp", "contact": contact_name, "text": message}
    )
    return {"queued_on_phone": f"WhatsApp to {contact_name}"}


def send_sms(contact_name: str, message: str) -> dict:
    """Send a text message (SMS) to one of the user's contacts.

    Args:
        contact_name: The person's name as the user says it. The phone looks
            the number up itself.
        message: What to say to them.
    """
    _queue_phone({"action": "sms", "contact": contact_name, "text": message})
    return {"queued_on_phone": f"SMS to {contact_name}"}


def call_contact(contact_name: str) -> dict:
    """Ring one of the user's contacts on the phone.

    Args:
        contact_name: The person's name as the user says it.
    """
    _queue_phone({"action": "call", "contact": contact_name})
    return {"queued_on_phone": f"calling {contact_name}"}


def navigate_to(destination: str) -> dict:
    """Start turn-by-turn navigation on the phone.

    Args:
        destination: Where to go, e.g. "Pune railway station" or "home".
    """
    _queue_phone({"action": "navigate", "destination": destination})
    return {"queued_on_phone": f"navigating to {destination}"}


def play_music(query: str) -> dict:
    """Play music on the phone.

    Args:
        query: A song, artist or playlist, e.g. "Kishore Kumar" or "lofi beats".
    """
    _queue_phone({"action": "play_music", "query": query})
    return {"queued_on_phone": f"playing {query}"}


# ---------- PC (laptop) tools ----------

def open_app_on_pc(app_name: str) -> dict:
    """Open an application on the user's PC (laptop).

    Args:
        app_name: One of: notepad, calculator, paint, chrome, edge, spotify,
            file explorer, settings, camera, vs code, android studio, word,
            excel, powerpoint.
    """
    return _queue_pc({"action": "open_app", "app": app_name}, f"open {app_name}")


def open_website_on_pc(url: str) -> dict:
    """Open a website in the PC's (laptop's) default browser.

    Args:
        url: The website address, e.g. "youtube.com".
    """
    return _queue_pc({"action": "open_url", "url": url}, f"open {url}")


def set_pc_volume(level: int) -> dict:
    """Set the PC's master speaker volume.

    Args:
        level: Volume percentage from 0 (mute) to 100 (max).
    """
    return _queue_pc({"action": "volume", "level": int(level)}, f"volume {level}")


def lock_pc() -> dict:
    """Lock the PC screen (like pressing Win+L)."""
    return _queue_pc({"action": "lock"}, "lock the PC")


def take_pc_screenshot() -> dict:
    """Take a screenshot on the PC and save it to its Pictures folder."""
    return _queue_pc({"action": "screenshot"}, "take a screenshot")
