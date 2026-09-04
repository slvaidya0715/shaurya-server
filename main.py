"""Shaurya's cloud brain.

Local:  uvicorn main:app --host 0.0.0.0 --port 8000
Cloud:  the Dockerfile runs it on port 7860 (Hugging Face Spaces).
"""

import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import actions
import brain
import storage

# Shared secret so a public URL isn't an open invitation to use your quota.
TOKEN = os.getenv("SHAURYA_TOKEN", "").strip()

app = FastAPI(title="Shaurya")


@app.on_event("startup")
def _warm_up() -> None:
    """Free hosting sleeps when idle; get the slow work out of the way as soon
    as we wake, so the first question isn't the one that pays for it."""
    import threading

    threading.Thread(target=brain.warm_up, daemon=True).start()

APK_PATH = Path(__file__).resolve().parent / "app-debug.apk"


def _check(token: str | None) -> None:
    if TOKEN and token != TOKEN:
        raise HTTPException(status_code=401, detail="Bad or missing token")


class ChatRequest(BaseModel):
    message: str
    device: str = "phone"  # "phone" or "pc" — which device the user spoke from


class ChatResponse(BaseModel):
    reply: str
    phone_actions: list = []


@app.get("/")
def health():
    return {
        "status": "ok",
        "brain_connected": brain.is_connected(),
        "cloud_storage": storage.using_cloud(),
        "pc_agent_online": actions.pc_agent_online(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_shaurya_token: str | None = Header(default=None)):
    _check(x_shaurya_token)
    result = brain.reply(req.message, req.device)
    return ChatResponse(reply=result["reply"], phone_actions=result["phone_actions"])


@app.get("/pc/poll")
def pc_poll(x_shaurya_token: str | None = Header(default=None)):
    """The laptop agent calls this on a loop to collect work."""
    _check(x_shaurya_token)
    return {"actions": actions.take_pc_actions()}


@app.get("/diag")
def diagnose(x_shaurya_token: str | None = Header(default=None)):
    """Where is the time going? Separates slow CPU from a slow network."""
    _check(x_shaurya_token)
    import time

    import requests as rq

    results = {}

    start = time.time()
    total = sum(i * i for i in range(2_000_000))  # pure CPU work
    results["cpu_2m_loop_s"] = round(time.time() - start, 2)

    start = time.time()
    try:
        rq.get("https://wttr.in/London?format=j1", headers={"User-Agent": "curl/8"}, timeout=20)
        results["https_wttr_s"] = round(time.time() - start, 2)
    except Exception as exc:
        results["https_wttr_s"] = f"failed: {exc}"

    start = time.time()
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            http_options=types.HttpOptions(timeout=60000),
        )
        client.models.generate_content(model="gemini-3.5-flash-lite", contents="hi")
        results["gemini_minimal_s"] = round(time.time() - start, 2)
    except Exception as exc:
        results["gemini_minimal_s"] = f"failed: {str(exc)[:120]}"

    return results


@app.post("/reset")
def reset(x_shaurya_token: str | None = Header(default=None)):
    _check(x_shaurya_token)
    brain.reset()
    return {"status": "conversation cleared"}


@app.get("/apk")
def download_apk():
    """Install the phone app from anywhere: open this URL on the phone."""
    if not APK_PATH.exists():
        return {"error": "APK not uploaded to this server"}
    return FileResponse(
        APK_PATH,
        media_type="application/vnd.android.package-archive",
        filename="Shaurya.apk",
    )
