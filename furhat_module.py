"""
Furhat wrapper — supports both Text-mode and WAV/URL-mode + speaking Event.

Public API (snake_case):
- say_text(text, blocking=False): speak with Furhat's internal TTS (Text-mode).
- say_url(url, blocking=False, lipsync=False): play external WAV by URL (WAV-mode).
- say_text_async(text): run say_text(blocking=True) in a background thread + speaking_done_event()
- say_url_async(url, lipsync=False): run say_url(blocking=True) in a background thread + speaking_done_event()
- stop_text(barge_in=True): stop speech started via say_text/say_text_async
- stop_url(barge_in=True): stop speech started via say_url/say_url_async
- speaking_done_event(): threading.Event set when Furhat finishes (natural or stopped)
- set_led(color), attend(...), gaze_yield(), gaze_hold(), set_voice(name), gesture(...)
- ping(timeout=1.5): health check.

Private helper:
- _sdk_stop_speaking(): unified low-level stop via SDK, with HTTP fallback.
"""
from __future__ import annotations
from typing import Any, Optional
import logging, time, requests, threading
from furhat_remote_api import FurhatRemoteAPI
import config
import random
# ============================== SHARED / CONFIG ==============================
BASE_HTTP   = f"http://{config.FURHAT_HOST}:{config.FURHAT_PORT}"
BASE_FURHAT = f"{BASE_HTTP}/furhat"

# IMPORTANT: pass ONLY the hostname to the SDK (it manages the port itself)
_host = str(config.FURHAT_HOST)
if ":" in _host:
    _host = _host.split(":")[0]

_client = FurhatRemoteAPI(_host)
_HTTP   = requests.Session()

TIMEOUTS = {"stop": 2.0, "ping": 1.5}

log = logging.getLogger("furhat")

# =============== SPEAKING EVENT (END-OF-SPEECH SIGNAL) ======================
# Set when Furhat's speech finishes (natural end or after stop_*()).
_speaking_done_evt = threading.Event()
_speaking_thread: Optional[threading.Thread] = None

def speaking_done_event() -> threading.Event:
    """Event set when Furhat has finished speaking (natural or stopped)."""
    return _speaking_done_evt

def _mark_speaking_started():
    """Clear the speaking event at the start of a new speech turn."""
    _speaking_done_evt.clear()

def _mark_speaking_finished():
    """Set the speaking event when a speech turn ends (natural/stop/error)."""
    _speaking_done_evt.set()

# ================================ CORE SPEECH ================================
def say_text(text: str, blocking: bool = False) -> Any:
    """
    Speak plain text via Furhat internal TTS (Text-mode).
    NOTE: No automatic LED or gaze changes here; controllers own that behavior.
    """
    return _client.say(text=text, blocking=blocking)

def say_url(url: str, blocking: bool = False, lipsync: bool = False) -> Any:
    """
    Play external audio file by URL (WAV-mode).
    NOTE: No automatic LED or gaze changes here; controllers own that behavior.
    """
    ts = int(time.time() * 1000)
    final_url = f"{url}?v={ts}"
    return _client.say(url=final_url, lipsync=lipsync, blocking=blocking)

# -------------------------- ASYNC CONVENIENCE -------------------------------
def say_text_async(text: str) -> None:
    global _speaking_thread
    _mark_speaking_started()
    def _run():
        try:
            say_text(text, blocking=True)
        except Exception:
            log.exception("say_text_async: error while speaking")
        finally:
            # No automatic LED/gaze here; controllers handle turn-end.
            _mark_speaking_finished()
    _speaking_thread = threading.Thread(target=_run, name="FurhatSayText", daemon=True)
    _speaking_thread.start()

def say_url_async(url: str, lipsync: bool = False) -> None:
    global _speaking_thread
    _mark_speaking_started()
    def _run():
        try:
            say_url(url, blocking=True, lipsync=lipsync)
        except Exception:
            log.exception("say_url_async: error while playing url")
        finally:
            # No automatic LED/gaze here; controllers handle turn-end.
            _mark_speaking_finished()
    _speaking_thread = threading.Thread(target=_run, name="FurhatSayURL", daemon=True)
    _speaking_thread.start()

# ================================ STOP HELPERS ===============================
def _sdk_stop_speaking():
    """Low-level stop of current speech via SDK, with HTTP fallback."""
    if hasattr(_client, "stop_speaking"):
        return _client.stop_speaking()
    r = _HTTP.post(f"{BASE_FURHAT}/say/stop", timeout=TIMEOUTS["stop"])
    r.raise_for_status()
    return r.json() if r.text else {"status": "ok"}

def stop_text(barge_in: bool = True) -> None:
    """
    Stop speech started by say_text/say_text_async.
    NOTE: No automatic LED or gaze changes here; controllers own that behavior.
    """
    try:
        _sdk_stop_speaking()
    except Exception:
        if barge_in:
            log.exception("stop_text(): stop failed (swallowed in barge-in)")
        else:
            raise
    _mark_speaking_finished()  # ensure controllers see end immediately

def stop_url(barge_in: bool = True) -> None:
    """
    Stop speech started by say_url/say_url_async (WAV playback).
    Semantics identical to stop_text regarding error handling.
    """
    try:
        _sdk_stop_speaking()
    except Exception:
        if barge_in:
            log.exception("stop_url(): stop failed (swallowed in barge-in)")
        else:
            raise
    _mark_speaking_finished()

# ============================== DEVICE / EXPRESSIONS =========================
def set_led(color: str):
    """Set LED to a named color; allowed: 'red' (speaking) or 'off' (listening)."""
    color_lc = (color or "").strip().lower()
    if color_lc not in ("red", "off"):
        raise ValueError("set_led: allowed colors are only 'red' and 'off'")
    r, g, b = (255, 0, 0) if color_lc == "red" else (0, 0, 0)
    return _client.set_led(red=r, green=g, blue=b)

def attend(x: float = 0.0, y: float = 0.0, z: float = 0.6):
    """Direct gaze to a 3D location (meters)."""
    # log.debug(f"[GAZE] attend({x:.2f}, {y:.2f}, {z:.2f})")  # enable if needed
    return _client.attend(location=f"{x},{y},{z}")

def gaze_yield():
    """Look directly at the user (yielding the floor)."""
    return attend(0.0, 0.0, 0.6)

def gaze_hold():
    """Look slightly away (holding the floor) in a random diagonal direction."""
    # Choose randomly: left/right × up/down
    x = random.choice([-0.06, 0.06])
    y = random.choice([-0.02, 0.02])
    return attend(x, y, 0.6)

def set_voice(name: str):
    """Switch the built-in Furhat voice (optional in Text-mode)."""
    return _client.set_voice(name=name)

def gesture(name: str | None = None, body: dict | None = None):
    """Perform a named gesture or custom body gesture."""
    if name:
        return _client.gesture(name=name)
    if body:
        return _client.gesture(body=body)
    return {"status": "ok"}

# ================================== HEALTH ==================================
def ping(timeout: float = TIMEOUTS["ping"]) -> bool:
    """Health check against the Remote API root."""
    try:
        r = _HTTP.get(BASE_FURHAT, timeout=timeout)
        return r.ok
    except Exception:
        return False

# ================================ INIT PROBE =================================
try:
    print("[FURHAT] Ready." if ping() else "[FURHAT] Not reachable.")
except Exception:
    print("[FURHAT] Init probe skipped.")
