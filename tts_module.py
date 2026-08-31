# tts_module.py — OpenAI streaming → optional WAV + 16 kHz mirror for VAP
from __future__ import annotations
import os, time, threading
from typing import Optional, Tuple
from collections import deque

import numpy as np
import soundfile as sf

try:
    import soxr  # high-quality resampler (optional)
except Exception:
    soxr = None

from openai import OpenAI
import config

# ===================== CONFIG =====================
TEMP_DIR = os.path.abspath(getattr(config, "TEMP_AUDIO_DIR", "temp_audio"))
os.makedirs(TEMP_DIR, exist_ok=True)

OAI_KEY    = getattr(config, "OPENAI_API_KEY", "")
TTS_MODEL  = getattr(config, "OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE  = getattr(config, "OPENAI_TTS_VOICE", "alloy")

# We assume OpenAI TTS PCM is 24 kHz; don't request a specific SR.
REQUEST_SAMPLE_RATE = 0
SRC_DEFAULT_SR      = 24000  # OpenAI PCM default

DST_SR   = 16000  # VAP target SR
FRAME_MS = int(getattr(config, "TTS_FRAME_MS", 20))
DST_SPF  = DST_SR * FRAME_MS // 1000  # 320 samples @16k/20ms

# Ring buffer capacity (~seconds of audio mirrored for VAP)
RING_CAP_SEC = float(getattr(config, "TTS_RING_CAP_SEC", 10.0))
RING_CAP_FR  = int(round((RING_CAP_SEC * DST_SR) / DST_SPF))

# Stream timeout: if no bytes for this long, abort turn
STREAM_TIMEOUT_S = float(getattr(config, "TTS_STREAM_TIMEOUT_S", 4.0))

# WAV paths (kept for future use)
BASE_NAME      = getattr(config, "TTS_FIXED_BASENAME", "robot_turn")
HTTP_BASE      = str(getattr(config, "LOCAL_HTTP_BASE", "http://127.0.0.1:8008"))
WAV_PART_PATH  = os.path.join(TEMP_DIR, f"{BASE_NAME}.part.wav")
WAV_FINAL_PATH = os.path.join(TEMP_DIR, f"{BASE_NAME}.wav")
WAV_URL        = f"{HTTP_BASE}/{BASE_NAME}.wav"

# ===================== OPENAI CLIENT (singleton) =====================
_client_singleton: Optional[OpenAI] = None
_client_lock = threading.Lock()

def _oai() -> OpenAI:
    global _client_singleton
    if _client_singleton:
        return _client_singleton
    if not (isinstance(OAI_KEY, str) and len(OAI_KEY) >= 10):
        raise RuntimeError("OPENAI_API_KEY missing/invalid in config.py")
    with _client_lock:
        if _client_singleton is None:
            _client_singleton = OpenAI(api_key=OAI_KEY)
    return _client_singleton

# ===================== RING (16 kHz mono int16) =====================
_ring_lock   = threading.Lock()
_ring_blocks: "deque[np.ndarray]" = deque()

def _ring_clear() -> None:
    with _ring_lock:
        _ring_blocks.clear()

def _ring_push_blocks(i16: np.ndarray, set_seen_evt: bool) -> None:
    if i16 is None or i16.size == 0:
        return
    x = np.asarray(i16, dtype=np.int16).ravel()
    n_frames = x.size // DST_SPF
    if n_frames <= 0:
        return
    x = x[: n_frames * DST_SPF]
    with _ring_lock:
        for i in range(n_frames):
            _ring_blocks.append(x[i*DST_SPF:(i+1)*DST_SPF])
        # trim to capacity
        while len(_ring_blocks) > RING_CAP_FR:
            _ring_blocks.popleft()
    if set_seen_evt:
        _vap_seen_evt.set()

def _ring_tail(seconds: float) -> np.ndarray:
    want = int(max(1, round(float(seconds) * DST_SR)))
    with _ring_lock:
        if not _ring_blocks:
            return np.zeros(want, dtype=np.int16)
        acc = 0
        sel = []
        for blk in reversed(_ring_blocks):
            sel.append(blk); acc += blk.size
            if acc >= want: break
        tail = np.concatenate(list(reversed(sel)), axis=0)
    return tail[-want:].astype(np.int16, copy=False)

# ===================== EVENTS / STATE =====================
_vap_seen_evt  = threading.Event()  # set on first mirrored 20ms block
_wav_ready_evt = threading.Event()  # set when final WAV moved into place
_cancel_evt    = threading.Event()  # barge-in / stop flag
_done_evt      = threading.Event()  # turn finished (success/cancel/error)

def vap_seen_event() -> threading.Event: return _vap_seen_evt
def done_event()    -> threading.Event: return _done_evt
def get_ready_url() -> str: return WAV_URL

def get_robot_audio(seconds: float) -> Tuple[np.ndarray, int]:
    return _ring_tail(seconds), DST_SR

def _reset_turn_state() -> None:
    _cancel_evt.clear(); _vap_seen_evt.clear(); _wav_ready_evt.clear(); _done_evt.clear()
    _ring_clear()
    try:
        if os.path.exists(WAV_PART_PATH):
            os.remove(WAV_PART_PATH)
    except Exception:
        pass

# ===================== HELPERS =====================
def _resample_to_16k_i16(x_i16: np.ndarray, src_sr: int) -> np.ndarray:
    if x_i16.size == 0:
        return x_i16
    if src_sr == DST_SR:
        return x_i16.astype(np.int16, copy=False)
    x_f32 = x_i16.astype(np.float32) / 32768.0
    if soxr is not None:
        y = soxr.resample(x_f32, src_sr, DST_SR).astype(np.float32, copy=False)
    else:
        # fallback linear interpolation
        n_out = int(round(x_f32.size * (DST_SR / float(src_sr))))
        if n_out <= 0:
            return np.zeros(0, dtype=np.int16)
        xp = np.linspace(0.0, 1.0, num=x_f32.size, endpoint=False, dtype=np.float32)
        xnew = np.linspace(0.0, 1.0, num=n_out,      endpoint=False, dtype=np.float32)
        y = np.interp(xnew, xp, x_f32).astype(np.float32, copy=False)
    return np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16, copy=False)

def warm_up_tts() -> None:
    """Optional warm-up (ignores errors)."""
    try:
        # No sample_rate parameter — stream a tiny response to prime the pipe.
        req = dict(model=TTS_MODEL, voice=TTS_VOICE, input="warm up", response_format="pcm")
        with _oai().audio.speech.with_streaming_response.create(**req) as resp:
            _ = next(resp.iter_bytes(), None)
    except Exception:
        pass

def stop_tts() -> None:
    _cancel_evt.set()
    _ring_clear()
    _done_evt.set()

# ===================== CORE =====================
def tts_start(text: str, make_wav: bool = True, blocking: bool = False) -> None:
    if blocking:
        _tts_worker(text, make_wav)
    else:
        threading.Thread(target=_tts_worker, args=(text, make_wav), name="TTSWorker", daemon=True).start()

def _tts_worker(text: str, make_wav: bool) -> None:
    t = (text or "").strip()
    if not t:
        _done_evt.set(); return

    _reset_turn_state()

    # WAV writer (optional)
    wf = None
    if make_wav:
        try:
            wf = sf.SoundFile(WAV_PART_PATH, mode="w", samplerate=DST_SR, channels=1, subtype="PCM_16")
        except Exception:
            wf = None

    # Always treat incoming PCM as 24 kHz
    src_sr = SRC_DEFAULT_SR  # 24000
    src_spf = src_sr * FRAME_MS // 1000
    frame_bytes = 2 * src_spf  # int16 @ 24k, 20 ms

    pcm_bytes = bytearray()
    first_frame_sent = False
    last_byte_t = time.monotonic()

    def _drain_frames_assuming(sr: int) -> int:
        """Drain full 20ms frames assuming source SR=sr. Returns number of 20ms frames pushed."""
        nonlocal first_frame_sent
        spf = sr * FRAME_MS // 1000
        fb  = 2 * spf
        pushed = 0
        while len(pcm_bytes) >= fb:
            # copy bytes before resizing
            block = bytes(pcm_bytes[:fb])
            del pcm_bytes[:fb]

            x_i16 = np.frombuffer(block, dtype="<i2")  # int16 LE @ sr
            i16   = _resample_to_16k_i16(x_i16, sr)    # -> int16 @ 16k
            if i16.size < DST_SPF:
                continue
            n = (i16.size // DST_SPF) * DST_SPF
            chunk = i16[:n]
            _ring_push_blocks(chunk, set_seen_evt=not first_frame_sent)
            first_frame_sent = True
            if wf is not None:
                try: wf.write(chunk)
                except Exception: pass
            pushed += n // DST_SPF
        return pushed

    # Build request (no sample_rate)
    req = dict(model=TTS_MODEL, voice=TTS_VOICE, input=t, response_format="pcm")

    try:
        with _oai().audio.speech.with_streaming_response.create(**req) as resp:
            for chunk in resp.iter_bytes():
                if _cancel_evt.is_set():
                    break
                if chunk:
                    pcm_bytes.extend(chunk)
                    last_byte_t = time.monotonic()
                    _drain_frames_assuming(src_sr)

                # timeout if stalled
                if (time.monotonic() - last_byte_t) > STREAM_TIMEOUT_S:
                    raise TimeoutError("TTS stream timeout (no new bytes)")

        # flush any remainder (best-effort)
        if not _cancel_evt.is_set() and len(pcm_bytes) >= 2:
            rem = (len(pcm_bytes) // 2) * 2
            block = bytes(pcm_bytes[:rem])  # copy, then shrink
            del pcm_bytes[:rem]
            x_i16 = np.frombuffer(block, dtype="<i2")
            i16   = _resample_to_16k_i16(x_i16, src_sr)
            if i16.size:
                if wf is not None:
                    try: wf.write(i16)
                    except Exception: pass
                # push only full frames to mirror
                n = (i16.size // DST_SPF) * DST_SPF
                if n > 0:
                    _ring_push_blocks(i16[:n], set_seen_evt=not first_frame_sent)
                    first_frame_sent = True

        # finalize WAV
        if make_wav and wf is not None:
            try:
                wf.flush(); wf.close()
                os.replace(WAV_PART_PATH, WAV_FINAL_PATH)
                _wav_ready_evt.set()
            except Exception:
                try:
                    wf.close()
                except Exception:
                    pass
                try:
                    if os.path.exists(WAV_PART_PATH):
                        os.remove(WAV_PART_PATH)
                except Exception:
                    pass

    except Exception as e:
        # Log minimally; do not fake 'seen'
        print(f"[TTS] stream error: {e}")
        try:
            if wf is not None:
                try: wf.close()
                except Exception: pass
                if os.path.exists(WAV_PART_PATH):
                    os.remove(WAV_PART_PATH)
        except Exception:
            pass
        _ring_clear()
        _done_evt.set()
        return

    _done_evt.set()
