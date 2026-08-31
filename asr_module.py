"""
ASR module — streaming microphone input + AssemblyAI partial/final transcripts.

Thesis-facing design goals:
- One microphone stream is shared by ASR and VAP.
- The VAP ring buffer keeps receiving microphone audio even when ASR output is paused.
- Partial and final transcripts are delivered through a non-blocking result queue.
- A lightweight energy-based speaking proxy is exposed so baseline and multi-agent
  controllers can use the SAME acoustic-end heuristic.
- Queue drops and backend errors are counted instead of being silently hidden.

Important limitation:
`pause(True)` still stops audio upload to AssemblyAI. This avoids robot self-transcription,
but it can lose the beginning of a user's barge-in. The later turn-taking redesign should
handle that explicitly (e.g. pre-roll replay/echo-aware ASR) rather than hiding the issue.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from assemblyai.streaming.v3 import (
    SpeechModel,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
)

import config

log = logging.getLogger("asr")

# =============================================================================
# Module state
# =============================================================================

_running: bool = False
_ASR_PAUSED: bool = False

_client: Optional[StreamingClient] = None
_stream_thread: Optional[threading.Thread] = None
_sd_stream: Optional[sd.InputStream] = None

_audio_q: Queue[Optional[bytes]] = Queue(maxsize=64)
_result_q: Queue[Tuple[str, str]] = Queue(maxsize=64)

_RING_SR = int(getattr(config, "SAMPLE_RATE", 16000))
_RING_MAX_SEC = float(getattr(config, "ASR_RING_MAX_SEC", 8.0))
_RING_MAX_SAMPLES = max(1, int(round(_RING_SR * _RING_MAX_SEC)))
_ring: deque[np.ndarray] = deque()
_ring_len_samples: int = 0
_ring_lock = threading.Lock()

_PARTIAL_DEBOUNCE_MS = int(getattr(config, "ASR_PARTIAL_DEBOUNCE_MS", 100))
_PARTIAL_FRESH_MS = int(getattr(config, "PARTIAL_FRESH_MS", 180))
# Normalised RMS amplitude. This is an ENERGY GATE, not a learned VAD.
_VAD_RMS_THRESHOLD = float(getattr(config, "VAD_RMS_THRESHOLD", 0.012))

_last_partial_text: str = ""
_last_partial_emit_ms: int = 0
_last_partial_received_ms: int = 0
_last_audio_callback_ms: int = 0
_last_rms_level: float = 0.0

_audio_drop_count: int = 0
_result_drop_count: int = 0
_stream_error_count: int = 0


# =============================================================================
# Helpers
# =============================================================================

def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalised_rms(i16: np.ndarray) -> float:
    """Return RMS amplitude in approximately [0, 1] for int16 mono audio."""
    if i16 is None or i16.size == 0:
        return 0.0
    x = i16.astype(np.float32, copy=False) / 32768.0
    return float(math.sqrt(float(np.mean(x * x))))


def _put_result(kind: str, text: str) -> None:
    global _result_drop_count
    try:
        _result_q.put_nowait((kind, text))
    except Full:
        _result_drop_count += 1
        log.warning("ASR result queue full; dropped %s result (drops=%d)", kind, _result_drop_count)


# =============================================================================
# Microphone callback (real-time)
# =============================================================================

def _sd_callback(indata, frames, time_info, status) -> None:
    """Capture one microphone block.

    The callback must remain fast: copy the block, update the VAP ring/energy
    state, and enqueue bytes for the background ASR worker.
    """
    global _ring_len_samples, _last_audio_callback_ms, _last_rms_level
    global _audio_drop_count

    if status:
        # Do not raise from a realtime callback; record the status instead.
        log.warning("sounddevice status: %s", status)

    blk = np.asarray(indata.reshape(-1), dtype=np.int16, order="C").copy()
    _last_audio_callback_ms = _now_ms()
    _last_rms_level = _normalised_rms(blk)

    # VAP ring is always filled, including while ASR upload is paused.
    with _ring_lock:
        _ring.append(blk)
        _ring_len_samples += int(blk.size)
        while _ring_len_samples > _RING_MAX_SAMPLES and _ring:
            oldest = _ring.popleft()
            _ring_len_samples -= int(oldest.size)

    # AssemblyAI upload is muted while paused.
    if _running and not _ASR_PAUSED:
        try:
            _audio_q.put_nowait(blk.tobytes())
        except Full:
            _audio_drop_count += 1
            # Logging every callback would itself hurt realtime performance.
            if _audio_drop_count == 1 or _audio_drop_count % 25 == 0:
                log.warning("ASR audio queue full; dropped blocks=%d", _audio_drop_count)


# =============================================================================
# AssemblyAI event handlers
# =============================================================================

def _on_turn(self: StreamingClient, event: TurnEvent) -> None:
    global _last_partial_text, _last_partial_emit_ms, _last_partial_received_ms

    if _ASR_PAUSED:
        return

    txt = (event.transcript or "").strip()
    if not txt:
        return

    now = _now_ms()

    if not getattr(event, "end_of_turn", False):
        _last_partial_received_ms = now

        # Debounce identical repeated partial hypotheses.
        if txt == _last_partial_text and (now - _last_partial_emit_ms) < _PARTIAL_DEBOUNCE_MS:
            return

        _last_partial_text = txt
        _last_partial_emit_ms = now
        _put_result("partial", txt)
        return

    # Final transcript.
    _put_result("final", txt)
    _last_partial_text = ""
    _last_partial_emit_ms = 0


def _on_error(self: StreamingClient, error: StreamingError) -> None:
    global _stream_error_count
    _stream_error_count += 1
    log.error("AssemblyAI streaming error (%d): %s", _stream_error_count, error)


# =============================================================================
# Streaming worker
# =============================================================================

def _audio_generator():
    while _running:
        try:
            item = _audio_q.get(timeout=0.03)
        except Empty:
            continue
        if item is None:
            break
        yield item


def _run_stream() -> None:
    global _stream_error_count
    try:
        if _client is None:
            return
        _client.stream(_audio_generator())
    except Exception as exc:
        _stream_error_count += 1
        log.exception("ASR streaming worker failed: %s", exc)
    finally:
        try:
            if _client is not None:
                _client.disconnect(terminate=True)
        except Exception:
            pass


# =============================================================================
# Public lifecycle API
# =============================================================================

def start_asr_stream(device: Optional[int] = None) -> None:
    """Start microphone capture and AssemblyAI streaming."""
    global _client, _stream_thread, _sd_stream, _running

    if _running:
        return

    api_key = (getattr(config, "ASSEMBLYAI_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is missing. Set it as an environment variable before starting ASR."
        )

    # Drain transient queues from a previous run.
    try:
        while True:
            _audio_q.get_nowait()
    except Empty:
        pass
    try:
        while True:
            _result_q.get_nowait()
    except Empty:
        pass

    dev = device if device is not None else getattr(config, "ASR_DEVICE_INDEX", None)
    blocksize = int(getattr(config, "ASR_BLOCKSIZE", int(_RING_SR * 0.05)))

    # Open the microphone BEFORE marking the module running. If this fails, the
    # state remains clean and the caller receives the real device error.
    try:
        _sd_stream = sd.InputStream(
            samplerate=_RING_SR,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            callback=_sd_callback,
            device=dev,
        )
        _sd_stream.start()
    except Exception:
        _sd_stream = None
        raise

    try:
        _client = StreamingClient(
            StreamingClientOptions(
                api_key=api_key,
                api_host="streaming.assemblyai.com",
            )
        )
        _client.on(StreamingEvents.Turn, _on_turn)
        _client.on(StreamingEvents.Error, _on_error)
        _client.connect(
            StreamingParameters(
                sample_rate=_RING_SR,
                speech_model=SpeechModel.universal_streaming_english,
            )
        )
    except Exception:
        try:
            _sd_stream.stop()
            _sd_stream.close()
        except Exception:
            pass
        _sd_stream = None
        _client = None
        raise

    _running = True
    _stream_thread = threading.Thread(target=_run_stream, name="ASRStream", daemon=True)
    _stream_thread.start()

    try:
        info = sd.query_devices(dev, "input") if dev is not None else sd.query_devices(kind="input")
        log.info(
            "ASR started: device=%s, samplerate=%d, blocksize=%d",
            info.get("name", dev) if isinstance(info, dict) else dev,
            _RING_SR,
            blocksize,
        )
    except Exception:
        log.info("ASR started: device=%s, samplerate=%d, blocksize=%d", dev, _RING_SR, blocksize)


def stop_asr_stream() -> None:
    global _running, _stream_thread, _sd_stream, _client

    _running = False
    try:
        _audio_q.put_nowait(None)
    except Full:
        pass

    try:
        if _sd_stream is not None:
            _sd_stream.stop()
            _sd_stream.close()
    except Exception:
        log.exception("Failed to close microphone stream cleanly")
    _sd_stream = None

    try:
        if _client is not None:
            _client.disconnect(terminate=True)
    except Exception:
        pass

    if _stream_thread is not None and _stream_thread.is_alive():
        _stream_thread.join(timeout=1.0)

    _stream_thread = None
    _client = None


def pause(flag: bool) -> None:
    """Pause/unpause ASR upload/emissions while leaving microphone/VAP capture alive."""
    global _ASR_PAUSED
    _ASR_PAUSED = bool(flag)


def is_paused() -> bool:
    return bool(_ASR_PAUSED)


def reset() -> None:
    """Clear transient transcript state; keep microphone and VAP ring alive."""
    global _last_partial_text, _last_partial_emit_ms
    _last_partial_text = ""
    _last_partial_emit_ms = 0
    try:
        while True:
            _result_q.get_nowait()
    except Empty:
        pass


# =============================================================================
# Transcript API
# =============================================================================

def has_new_result() -> bool:
    return not _result_q.empty()


def get_result() -> Optional[Tuple[str, str]]:
    try:
        return _result_q.get_nowait()
    except Empty:
        return None


def last_partial_time_ms() -> int:
    """Timestamp of the most recent partial ASR hypothesis received."""
    return int(_last_partial_received_ms)


# =============================================================================
# Lightweight acoustic state for simple baseline/multi-agent endpointing
# =============================================================================

def get_vad_level() -> float:
    """Return the latest normalised RMS microphone level.

    Despite the legacy name, this is an energy level, not a learned VAD score.
    """
    return float(_last_rms_level)


def user_speaking() -> bool:
    """Shared speaking proxy used by the controlled baseline/RQ2 conditions.

    A fresh ASR partial is strong evidence of ongoing speech. Otherwise fall
    back to a simple RMS energy threshold. Keep the threshold identical across
    experimental conditions and report/tune it during pilot testing.
    """
    now = _now_ms()
    if _last_partial_received_ms and (now - _last_partial_received_ms) <= _PARTIAL_FRESH_MS:
        return True
    return bool(_last_rms_level >= _VAD_RMS_THRESHOLD)


# =============================================================================
# VAP support: recent microphone tail at SAMPLE_RATE, int16 mono
# =============================================================================

def get_user_audio(seconds: float) -> np.ndarray:
    want = max(0.01, float(seconds))
    want_samples = int(min(_RING_MAX_SAMPLES, round(want * _RING_SR)))

    with _ring_lock:
        if _ring_len_samples <= 0:
            return np.zeros(int(_RING_SR * want), dtype=np.int16)

        collected = []
        acc = 0
        for blk in reversed(_ring):
            collected.append(blk)
            acc += int(blk.size)
            if acc >= want_samples:
                break

        if not collected:
            return np.zeros(int(_RING_SR * want), dtype=np.int16)

        tail = np.concatenate(list(reversed(collected)), axis=0)
        if tail.size > want_samples:
            tail = tail[-want_samples:]

    return tail


# =============================================================================
# Diagnostics for reproducibility / robustness reporting
# =============================================================================

def diagnostics() -> dict:
    return {
        "running": bool(_running),
        "paused": bool(_ASR_PAUSED),
        "sample_rate": int(_RING_SR),
        "ring_max_seconds": float(_RING_MAX_SEC),
        "vad_rms_threshold": float(_VAD_RMS_THRESHOLD),
        "last_rms_level": float(_last_rms_level),
        "last_audio_callback_ms": int(_last_audio_callback_ms),
        "last_partial_received_ms": int(_last_partial_received_ms),
        "audio_queue_drops": int(_audio_drop_count),
        "result_queue_drops": int(_result_drop_count),
        "stream_errors": int(_stream_error_count),
    }
