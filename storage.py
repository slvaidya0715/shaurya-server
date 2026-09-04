"""Storage that survives restarts, on a laptop or on a free cloud host.

Cloud hosts wipe their filesystem on every restart, so when HF_TOKEN and
HF_DATA_REPO are set we keep files in a private Hugging Face dataset instead.
Falls back to a local folder when running on the laptop.
"""

import base64
import hashlib
import json
import os
import threading
from pathlib import Path

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_DATA_REPO = os.getenv("HF_DATA_REPO", "").strip()  # e.g. "yourname/shaurya-data"

# Anything stored off this server is encrypted first, so whoever holds the
# disk — the storage provider included — sees only ciphertext.
_SECRET = os.getenv("SHAURYA_SECRET", "").strip()


def _cipher():
    if not _SECRET:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None
    key = base64.urlsafe_b64encode(hashlib.sha256(_SECRET.encode()).digest())
    return Fernet(key)


def _encrypt(text: str) -> bytes:
    box = _cipher()
    return box.encrypt(text.encode()) if box else text.encode()


def _decrypt(raw: bytes) -> str:
    box = _cipher()
    if box:
        try:
            return box.decrypt(raw).decode()
        except Exception:
            pass  # not encrypted yet (first run after turning this on)
    return raw.decode()

LOCAL_DIR = Path(__file__).resolve().parent / "data"
_lock = threading.Lock()
_cache: dict = {}


def using_cloud() -> bool:
    return bool(HF_TOKEN and HF_DATA_REPO)


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=HF_TOKEN)


def _ensure_repo() -> None:
    api = _api()
    api.create_repo(
        repo_id=HF_DATA_REPO, repo_type="dataset", private=True, exist_ok=True
    )


def load(name: str, default):
    """Read a JSON file by name, returning `default` if it doesn't exist yet."""
    with _lock:
        if name in _cache:
            return _cache[name]

        value = default
        if using_cloud():
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=HF_DATA_REPO,
                    filename=name,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                value = json.loads(_decrypt(Path(path).read_bytes()))
            except Exception:
                value = default  # not created yet, or offline
        else:
            path = LOCAL_DIR / name
            if path.exists():
                try:
                    value = json.loads(_decrypt(path.read_bytes()))
                except (json.JSONDecodeError, OSError, ValueError):
                    value = default

        _cache[name] = value
        return value


def save(name: str, obj) -> None:
    """Write a JSON file by name."""
    with _lock:
        _cache[name] = obj
        text = json.dumps(obj, indent=2, ensure_ascii=False)

        if using_cloud():
            try:
                _ensure_repo()
                _api().upload_file(
                    path_or_fileobj=_encrypt(text),
                    path_in_repo=name,
                    repo_id=HF_DATA_REPO,
                    repo_type="dataset",
                )
            except Exception as exc:
                print(f"[storage] cloud save failed: {exc}", flush=True)
        else:
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            (LOCAL_DIR / name).write_bytes(_encrypt(text))
