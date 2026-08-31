"""
furhat_realtime_module.py
=========================

Realtime Furhat adapter for the thesis experiments.

This module:
1. connects to Furhat's Realtime API;
2. speaks text with Furhat's own TTS;
3. records actual speak-start / speak-end event times;
4. captures the Furhat speaker channel as 16-kHz PCM16;
5. keeps recent speaker audio in a ring buffer for VAP;
6. exposes a small synchronous API similar to the legacy furhat_module.py.

Dependency:
    pip install furhat-realtime-api numpy
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

try:
    from furhat_realtime_api import AsyncFurhatClient, Events
except Exception as exc:
    AsyncFurhatClient = None
    Events = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

log = logging.getLogger("furhat_realtime")

HOST = "127.0.0.1"
SPEAKER_SAMPLE_RATE = 16_000
RING_SECONDS = 10.0
RING_MAX_SAMPLES = int(SPEAKER_SAMPLE_RATE * RING_SECONDS)

_client: Optional[Any] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_connected = False
_audio_capture_started = False

_speaking_started_evt = threading.Event()
_speaking_done_evt = threading.Event()

_last_speech_start_ms: Optional[int] = None
_last_speech_end_ms: Optional[int] = None
_last_gen_time_ms: Optional[float] = None
_last_spoken_text = ""
_last_speech_aborted = False
_last_speech_failed = False

_ring_lock = threading.Lock()
_ring = deque()
_ring_samples = 0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _event_get(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _ring_clear() -> None:
    global _ring_samples
    with _ring_lock:
        _ring.clear()
        _ring_samples = 0


def _ring_push(samples: np.ndarray) -> None:
    global _ring_samples
    if samples is None or samples.size == 0:
        return
    block = np.asarray(samples, dtype=np.int16).reshape(-1).copy()
    with _ring_lock:
        _ring.append(block)
        _ring_samples += block.size
        while _ring_samples > RING_MAX_SAMPLES and _ring:
            old = _ring.popleft()
            _ring_samples -= old.size


def get_robot_audio(seconds: float = 1.0) -> np.ndarray:
    wanted = max(
        1,
        min(RING_MAX_SAMPLES, int(round(float(seconds) * SPEAKER_SAMPLE_RATE))),
    )
    with _ring_lock:
        if not _ring:
            return np.zeros(wanted, dtype=np.int16)
        selected = []
        total = 0
        for block in reversed(_ring):
            selected.append(block)
            total += block.size
            if total >= wanted:
                break
        audio = np.concatenate(list(reversed(selected)))

    if audio.size < wanted:
        padded = np.zeros(wanted, dtype=np.int16)
        padded[-audio.size:] = audio
        return padded
    return audio[-wanted:].astype(np.int16, copy=False)


async def _on_speak_start(event: Any) -> None:
    global _last_speech_start_ms, _last_gen_time_ms, _last_spoken_text
    _last_speech_start_ms = _now_ms()
    _last_gen_time_ms = _event_get(event, "gen_time")
    _last_spoken_text = str(_event_get(event, "text", "") or "")
    _speaking_done_evt.clear()
    _speaking_started_evt.set()


async def _on_speak_end(event: Any) -> None:
    global _last_speech_end_ms, _last_spoken_text
    global _last_speech_aborted, _last_speech_failed
    _last_speech_end_ms = _now_ms()
    _last_spoken_text = str(_event_get(event, "text", _last_spoken_text) or "")
    _last_speech_aborted = bool(_event_get(event, "aborted", False))
    _last_speech_failed = bool(_event_get(event, "failed", False))
    _speaking_done_evt.set()


async def _on_audio_data(event: Any) -> None:
    encoded = _event_get(event, "speaker")
    if not encoded:
        return
    try:
        raw = base64.b64decode(encoded)
        samples = np.frombuffer(raw, dtype="<i2")
        _ring_push(samples)
    except Exception:
        log.exception("Could not decode Furhat speaker audio")


async def _async_connect(host: str) -> None:
    global _client, _connected, _audio_capture_started

    if AsyncFurhatClient is None:
        raise RuntimeError(
            "furhat-realtime-api is not installed or failed to import: "
            f"{_IMPORT_ERROR}"
        )

    if _connected and _client is not None:
        return

    _client = AsyncFurhatClient(host)
    _client.add_handler(Events.response_speak_start, _on_speak_start)
    _client.add_handler(Events.response_speak_end, _on_speak_end)
    _client.add_handler(Events.response_audio_data, _on_audio_data)

    await _client.connect()
    _connected = True

    await _client.request_audio_start(
        sample_rate=SPEAKER_SAMPLE_RATE,
        microphone=False,
        speaker=True,
    )
    _audio_capture_started = True


async def _async_disconnect() -> None:
    global _client, _connected, _audio_capture_started

    if _client is None:
        _connected = False
        _audio_capture_started = False
        return

    if _audio_capture_started:
        try:
            await _client.request_audio_stop()
        except Exception:
            pass

    try:
        await _client.disconnect()
    finally:
        _client = None
        _connected = False
        _audio_capture_started = False


def _loop_worker(host: str, ready_evt: threading.Event, error_box: list) -> None:
    global _loop
    loop = asyncio.new_event_loop()
    _loop = loop
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_async_connect(host))
    except Exception as exc:
        error_box.append(exc)
        ready_evt.set()
        loop.close()
        _loop = None
        return

    ready_evt.set()

    try:
        loop.run_forever()
    finally:
        if _connected:
            try:
                loop.run_until_complete(_async_disconnect())
            except Exception:
                pass
        loop.close()
        _loop = None


def connect(host: str = HOST, timeout: float = 8.0) -> bool:
    global _thread

    if _connected:
        return True

    ready_evt = threading.Event()
    error_box = []

    _thread = threading.Thread(
        target=_loop_worker,
        args=(host, ready_evt, error_box),
        name="FurhatRealtimeLoop",
        daemon=True,
    )
    _thread.start()

    if not ready_evt.wait(timeout=timeout):
        raise TimeoutError(
            f"Timed out connecting to Furhat Realtime API at {host}:9000"
        )

    if error_box:
        raise RuntimeError(f"Furhat Realtime connection failed: {error_box[0]}")

    return bool(_connected)


def _submit(coro, timeout: Optional[float] = None):
    if not _connected or _client is None or _loop is None:
        raise RuntimeError("Call connect() before using Furhat Realtime.")
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    if timeout is None:
        return fut
    return fut.result(timeout=timeout)


def disconnect(timeout: float = 5.0) -> None:
    global _thread
    if _loop is None:
        return

    try:
        _submit(_async_disconnect(), timeout=timeout)
    except Exception:
        pass

    loop = _loop
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)

    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=timeout)
    _thread = None


def ping() -> bool:
    return bool(_connected and _client is not None and _loop is not None)


def speaking_started_event() -> threading.Event:
    return _speaking_started_evt


def speaking_done_event() -> threading.Event:
    return _speaking_done_evt


def say_text_async(text: str, abort: bool = False) -> None:
    t = (text or "").strip()
    if not t:
        return

    _speaking_started_evt.clear()
    _speaking_done_evt.clear()
    _ring_clear()

    _submit(
        _client.request_speak_text(
            t,
            wait=False,
            abort=bool(abort),
        )
    )


def say_text(text: str, blocking: bool = False, timeout: float = 60.0) -> None:
    say_text_async(text)
    if blocking and not _speaking_done_evt.wait(timeout=timeout):
        raise TimeoutError("Timed out waiting for Furhat to finish speaking")


def stop_text(barge_in: bool = True, timeout: float = 5.0) -> None:
    if not _connected:
        return
    try:
        _submit(_client.request_speak_stop(), timeout=timeout)
    finally:
        if barge_in:
            _speaking_done_evt.set()


def last_speech_metrics() -> dict:
    duration_ms = None
    if _last_speech_start_ms is not None and _last_speech_end_ms is not None:
        duration_ms = max(0, _last_speech_end_ms - _last_speech_start_ms)

    return {
        "start_ms": _last_speech_start_ms,
        "end_ms": _last_speech_end_ms,
        "duration_ms": duration_ms,
        "gen_time_ms": _last_gen_time_ms,
        "text": _last_spoken_text,
        "aborted": _last_speech_aborted,
        "failed": _last_speech_failed,
    }


def set_led(color: str) -> None:
    mapping = {"red": "#FF0000", "off": "#000000"}
    c = (color or "").strip().lower()
    if c not in mapping:
        raise ValueError("set_led supports only 'red' and 'off'")
    _submit(_client.request_led_set(mapping[c]))


def attend(x: float = 0.0, y: float = 0.0, z: float = 0.6) -> None:
    _submit(_client.request_attend_location(float(x), float(y), float(z)))


def gaze_yield() -> None:
    attend(0.0, 0.0, 0.6)


def gaze_hold() -> None:
    # Fixed rather than random for experimental consistency.
    attend(0.06, 0.02, 0.6)
