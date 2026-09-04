"""Shaurya's brain: a Gemini chat session with tool calling.

We run the tool loop by hand — the SDK's automatic function calling silently
drops calls, which made the assistant claim it had done things it hadn't.
"""

import os

from dotenv import load_dotenv

import actions
from actions import (
    call_contact,
    lock_pc,
    navigate_to,
    open_app_on_pc,
    open_app_on_phone,
    open_website_on_pc,
    open_website_on_phone,
    play_music,
    send_sms,
    send_whatsapp,
    set_alarm,
    set_flashlight,
    set_pc_volume,
    set_timer,
    take_pc_screenshot,
)
from memory import forget, load_memories, remember
from tools import get_weather, web_search

load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Fastest first: a voice assistant is judged on how quickly it answers, and
# the lite model handles questions and tool calls in about two seconds while
# the full model takes several. The rest are fallbacks for when a free daily
# quota runs dry — each model has its own allowance.
# The lite models carry a far bigger free daily allowance than gemini-2.5-flash
# (which runs dry after about twenty questions), so they lead.
_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]
_model_index = 0

# Free models sometimes get badly congested — one measured 23 seconds for
# "hi" while its neighbours answered in 1.5. Rather than let the user wait,
# give up quickly and move to the next model in the list.
MODEL_TIMEOUT_MS = 12000

TOOLS = [
    get_weather,
    web_search,
    remember,
    forget,
    # Phone tools — executed by the Android app
    open_app_on_phone,
    open_website_on_phone,
    set_flashlight,
    set_alarm,
    set_timer,
    send_whatsapp,
    send_sms,
    call_contact,
    navigate_to,
    play_music,
    # PC tools — executed by the agent running on the laptop
    open_app_on_pc,
    open_website_on_pc,
    set_pc_volume,
    lock_pc,
    take_pc_screenshot,
]
TOOL_FUNCS = {f.__name__: f for f in TOOLS}
MAX_TOOL_ROUNDS = 6


def _system_prompt() -> str:
    memories = load_memories()
    memory_text = (
        "\n".join(f"- {m}" for m in memories) if memories else "(nothing saved yet)"
    )
    laptop = "connected" if actions.pc_agent_online() else "not connected right now"
    return (
        "You are Shaurya, a sharp, witty AI voice assistant in the spirit of "
        "Jarvis from Iron Man. Personality: confident, warm, lightly humorous, "
        "occasionally addressing the user as 'sir' — never robotic, never "
        "long-winded.\n\n"
        "HOW YOU ACT: You have real tools, and calling them is how you get "
        "things done. Tool calls are internal and invisible to the user — "
        "they are not part of your spoken reply, so use them freely and as "
        "often as needed. When the user asks you to do anything (open an app "
        "or website, change volume, flashlight, lock the PC, screenshot, "
        "remember or forget something) you MUST call the matching tool first, "
        "then speak. Stating that you did something without having called the "
        "tool is a lie and strictly forbidden. Use get_weather and web_search "
        "whenever a question needs fresh information.\n\n"
        "PEOPLE: to message or ring someone, pass the name the user said "
        "straight to the tool — send_whatsapp, send_sms or call_contact. The "
        "phone looks the person up in its own address book, so never ask for "
        "a phone number and never expect to see one. After sending a WhatsApp "
        "message, mention that it's typed out and waiting for them to hit "
        "send, because WhatsApp allows no way to press it for them. Texts and "
        "calls do go through on their own.\n\n"
        "TWO DEVICES: the user has a phone and a PC (laptop), and you have "
        "separate tools for each (..._on_phone vs ..._on_pc). Every user "
        "message is tagged with the device they are speaking from, like "
        "[from phone]. When they don't name a device, act on the device they "
        "are speaking from. When they say 'on the laptop', 'on the PC', 'on "
        "my phone' etc., use that device's tool instead. If a PC tool reports "
        "the agent isn't running, tell the user their laptop isn't connected "
        f"— right now the laptop is {laptop}.\n\n"
        "HOW YOU SPEAK: after any tool calls are done, your final text is "
        "read aloud, so keep it to one to three short conversational "
        "sentences — no markdown, no emojis, no URLs.\n\n"
        f"Things you remember about the user:\n{memory_text}"
    )


_client = None
_chat = None


def is_connected() -> bool:
    return bool(_API_KEY)


def _get_chat():
    global _client, _chat
    from google import genai
    from google.genai import types

    if _client is None:
        _client = genai.Client(
            api_key=_API_KEY,
            http_options=types.HttpOptions(timeout=MODEL_TIMEOUT_MS),
        )
    if _chat is None:
        model = _MODELS[_model_index]
        extra = {}
        # The 2.5 models think before answering, which costs seconds and can
        # eat the whole reply, leaving nothing to speak. Newer models don't
        # accept this setting at all, hence the check.
        if model.startswith("gemini-2.5"):
            extra["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        _chat = _client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=_system_prompt(),
                tools=TOOLS,
                **extra,
                # We run the tool loop ourselves — the SDK's automatic execution
                # proved unreliable (claimed success without running anything).
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    return _chat


# Anything the user might consider private stays out of the server logs.
_PRIVATE_ARGS = {"message", "text", "contact_name", "contact", "query", "fact"}


def _safe(args: dict) -> dict:
    """Log which tool ran and with what shape, never the personal content."""
    return {
        key: ("<hidden>" if key in _PRIVATE_ARGS else value)
        for key, value in args.items()
    }


def _pending_calls(response):
    parts = response.candidates[0].content.parts or []
    return [p.function_call for p in parts if getattr(p, "function_call", None)]


def _reply_once(message: str, device: str) -> str:
    import time

    from google.genai import types

    chat = _get_chat()
    started = time.time()

    def mark(label: str, since: float) -> None:
        print(f"[time] {label}: {time.time() - since:.2f}s", flush=True)

    turn = time.time()
    response = chat.send_message(f"[from {device}] {message}")
    mark(f"model({_MODELS[_model_index]})", turn)

    # Manual tool loop: execute every requested tool, feed results back,
    # repeat until the model answers with plain text.
    for _ in range(MAX_TOOL_ROUNDS):
        calls = _pending_calls(response)
        if not calls:
            break
        result_parts = []
        for call in calls:
            func = TOOL_FUNCS.get(call.name)
            args = dict(call.args or {})
            tool_started = time.time()
            try:
                result = func(**args) if func else {"error": f"unknown tool {call.name}"}
            except Exception as exc:
                result = {"error": str(exc)}
            mark(f"tool({call.name})", tool_started)
            print(f"[tool] {call.name}({_safe(args)})", flush=True)
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )
        turn = time.time()
        response = chat.send_message(result_parts)
        mark("model(after tools)", turn)

    mark("TOTAL", started)
    text = (response.text or "").strip()
    if not text:
        # A model that answers with nothing is no use; let the caller move on
        # to the next one rather than telling the user we came up empty.
        raise RuntimeError("empty response from model")
    return text


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def _should_try_next_model(exc: Exception) -> bool:
    text = str(exc)
    return (
        _is_quota_error(exc)
        or "NOT_FOUND" in text
        or "404" in text
        or "UNAVAILABLE" in text
        or "503" in text
        # A model that has stopped answering in time is no better than one
        # that's down, so treat slowness as a reason to switch too.
        or "DEADLINE_EXCEEDED" in text
        or "504" in text
        or "timeout" in text.lower()
        or "timed out" in text.lower()
        or "empty response from model" in text
    )


MAX_HISTORY_MESSAGES = 24


def _trim_history() -> None:
    """Every question resends the whole conversation, so an unbounded history
    makes the assistant steadily slower and burns the daily quota faster.
    Keep recent turns; the long-term facts live in the system prompt anyway."""
    global _chat
    if _chat is None:
        return
    try:
        history = _chat.get_history()
        if len(history) <= MAX_HISTORY_MESSAGES:
            return
        from google.genai import types

        kept = history[-(MAX_HISTORY_MESSAGES // 2):]
        model = _MODELS[_model_index]
        extra = {}
        if model.startswith("gemini-2.5"):
            extra["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        _chat = _client.chats.create(
            model=model,
            history=kept,
            config=types.GenerateContentConfig(
                system_instruction=_system_prompt(),
                tools=TOOLS,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                **extra,
            ),
        )
        print(f"[brain] trimmed history to {len(kept)} messages", flush=True)
    except Exception as exc:
        print(f"[brain] history trim skipped: {exc}", flush=True)


def reply(message: str, device: str = "phone") -> dict:
    """Returns {"reply": spoken text, "phone_actions": [...]} for the client."""
    global _model_index, _chat
    actions.start_request()
    if not _API_KEY:
        return {
            "reply": "My brain has no Gemini API key configured, sir.",
            "phone_actions": [],
        }
    while True:
        try:
            text = _reply_once(message, device)
            _trim_history()
            return {"reply": text, "phone_actions": actions.collect_phone_actions()}
        except Exception as exc:
            if _should_try_next_model(exc) and _model_index + 1 < len(_MODELS):
                _model_index += 1
                _chat = None  # fresh chat on the fallback model
                print(
                    f"[brain] switching model to {_MODELS[_model_index]} ({str(exc)[:80]})",
                    flush=True,
                )
                continue
            if _is_quota_error(exc):
                text = (
                    "I'm afraid I've used up today's free thinking quota, sir. "
                    "It resets tomorrow."
                )
            else:
                text = f"Sorry, my brain hit an error: {exc}"
            return {"reply": text, "phone_actions": []}


def reset() -> None:
    """Start a fresh conversation (also reloads memories into the prompt)."""
    global _chat
    _chat = None


def warm_up() -> None:
    """Do the slow first-time work at startup rather than on the first question:
    downloading saved memories and opening the connection to the model."""
    try:
        _get_chat()
        # Open the connections the tools will need, so the first real question
        # doesn't pay for DNS and TLS handshakes on a slow free instance.
        import requests

        for url in ("https://wttr.in/London?format=j1", "https://duckduckgo.com/"):
            try:
                requests.get(url, headers={"User-Agent": "curl/8"}, timeout=10)
            except Exception:
                pass
        print("[brain] warmed up and ready", flush=True)
    except Exception as exc:
        print(f"[brain] warm-up skipped: {exc}", flush=True)
