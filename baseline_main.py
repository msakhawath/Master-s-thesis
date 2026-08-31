# ============================================================
# baseline_main.py — Simplex (no barge-in) baseline (clean, gated)
# LLM is triggered ONLY after 1.0 s continuous silence confirmed.
# Latency anchors:
#   • Acoustic END = first silent frame (LAST_USER_STOP_MS)
#   • Confirmation = 1.0 s silence (USER_END_STAMPED + CSV user_end)
#   • ASR latency  = ASR_FINAL_MS − LAST_USER_STOP_MS
#   • FTO (gap)    = ROBOT_START_MS − LAST_USER_STOP_MS
# Robot start anchor (this build):
#   • Device playback onset, stamped immediately AFTER FR.say_text_async(...)
#   • BUT we start TTS first and wait for the FIRST AUDIO CHUNK before sending text,
#     so TTS generation latency is included in FTO/response-time logs.
# ============================================================

from __future__ import annotations
import time
import logging
from typing import List, Dict, Optional

import config
import asr_module as ASR
# Use the LOCAL Ollama model so the baseline runs on the SAME model as the
# multi-agent system (gemma3:1b). This isolates the user-study comparison to
# the agent ARCHITECTURE only — not the model or cloud-vs-local.
import llm_module_ollama as LLM
import furhat_module as FR
import logger  # SessionLogger
import tts_module as TTS  # <<< added: used only to wait until first audio chunk is ready

# ---------- Console logger ----------
log = logging.getLogger("baseline_controller")
log.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(_h)

# ========= Conversation + Timing State =========
HISTORY: List[Dict[str, str]] = []

LAST_USER_START_MS: Optional[int] = None
LAST_USER_STOP_MS: Optional[int] = None   # acoustic end (first silent frame)
ASR_FINAL_MS: Optional[int] = None
ROBOT_START_MS: Optional[int] = None
ROBOT_END_MS: Optional[int] = None

SILENCE_START_MS: Optional[int] = None
USER_END_STAMPED: bool = False
LAST_PARTIAL_SEEN_MS: Optional[int] = None

PREV_USER_SPEAKING: bool = False

START_TIME_S: Optional[float] = None
TIME_LIMIT_S: float = getattr(config, "SESSION_TIME_LIMIT_S", 600.0)

FINAL_SENTENCE: str = getattr(
    config, "FINAL_SENTENCE",
    "Thank you for your time. This concludes our interview."
)

SESSION_LOGGER: Optional[logger.SessionLogger] = None

# ========= Utilities =========
def now_ms() -> int:
    return int(time.time() * 1000)

def elapsed_s() -> float:
    return 0.0 if START_TIME_S is None else max(0.0, time.time() - START_TIME_S)

def _csv_log(**kwargs):
    if SESSION_LOGGER is not None:
        SESSION_LOGGER.log_event(**kwargs)

def _append_user(text: str, user_turn_len_ms: Optional[int], asr_final_latency_ms: Optional[int]) -> None:
    t = (text or "").strip()
    HISTORY.append({"role": "user", "content": t})
    log.info(f"[USER] {t}")
    _csv_log(event_type="user_final",
             user_text=t,
             user_turn_len_ms=user_turn_len_ms,
             asr_final_latency_ms=asr_final_latency_ms)

def _append_bot(full_text: str) -> None:
    t = (full_text or "").strip()
    HISTORY.append({"role": "assistant", "content": t})
    log.info(f"[BOT ] {t}")
    _csv_log(event_type="robot_end",
             bot_text=t,
             robot_turn_len_ms=(max(0, (ROBOT_END_MS or 0) - (ROBOT_START_MS or 0))
                                if (ROBOT_END_MS and ROBOT_START_MS) else None))

# ========= VAD helpers =========
def _detect_user_speaking_now() -> Optional[bool]:
    try:
        speaking = ASR.user_speaking()  # bool if available
        if isinstance(speaking, bool):
            return speaking
    except Exception:
        pass
    try:
        level = ASR.get_vad_level()
        if level is not None:
            thr = getattr(config, "VAD_LEVEL_THRESHOLD", 0.5)
            return bool(level > thr)
    except Exception:
        pass
    try:
        lp = ASR.last_partial_time_ms()
        if isinstance(lp, int):
            return (now_ms() - lp) <= getattr(config, "PARTIAL_FRESH_MS", 180)
    except Exception:
        pass
    return None

def _speaking_from_partials(nowt: int) -> Optional[bool]:
    if LAST_PARTIAL_SEEN_MS is None:
        return None
    return (nowt - LAST_PARTIAL_SEEN_MS) <= int(getattr(config, "PARTIAL_FRESH_MS", 180))

def _update_audio_state_on_tick() -> None:
    global PREV_USER_SPEAKING, LAST_USER_START_MS
    speaking_now = _detect_user_speaking_now()
    if speaking_now is None:
        return
    if not PREV_USER_SPEAKING and speaking_now and LAST_USER_START_MS is None:
        LAST_USER_START_MS = now_ms()
    PREV_USER_SPEAKING = speaking_now

# ========= Lifecycle =========
def init_system() -> None:
    global SESSION_LOGGER, START_TIME_S
    SESSION_LOGGER = logger.SessionLogger(run_mode="baseline")
    _csv_log(event_type="session_start")
    START_TIME_S = time.time()
    # Warm the local model into RAM before the interview, so the first turn
    # does not pay the one-time cold-load (~tens of seconds on CPU).
    if hasattr(LLM, "warmup"):
        try:
            log.info("Warming up local model (loading into RAM)...")
            LLM.warmup()
        except Exception:
            pass
    ASR.start_asr_stream()
    ASR.pause(False)
    try: FR.set_led("off")
    except Exception: pass
    log.info("System initialized")

def shutdown_system() -> None:
    try: FR.stop_text(barge_in=True)
    except Exception: pass
    try: ASR.stop_asr_stream()
    except Exception: pass
    if SESSION_LOGGER is not None:
        try: SESSION_LOGGER.close()
        except Exception: pass
    log.info("System shutdown complete")

# ========= Speak (Simplex) — include TTS first-chunk latency before device onset =========
def _speak_baseline_chunks(chunks: List[str]) -> None:
    """
    Speak the provided chunks on the Furhat device.
    Modified baseline: start TTS first, wait until the FIRST audio chunk is produced (no VAP logic),
    then send the text to Furhat and anchor robot start immediately AFTER say_text_async(...).
    This way, the measured FTO includes the TTS generation latency up to the first chunk.
    """
    global ROBOT_START_MS, ROBOT_END_MS
    ROBOT_START_MS = None
    ROBOT_END_MS = None

    # Pause ASR during robot output
    try:
        ASR.pause(True)
        ASR.reset()
    except Exception:
        pass
    try:
        FR.set_led("red")
    except Exception:
        pass

    if not chunks:
        try: FR.set_led("off")
        except Exception: pass
        try: FR.gaze_yield()
        except Exception: pass
        try:
            ASR.reset()
            ASR.pause(False)
        except Exception:
            pass
        return

    try: FR.gaze_hold()
    except Exception: pass

    # ---- Start TTS generation (no streaming to Furhat) and wait for the first audio chunk
    tts_call_start_ms = now_ms()
    full_text = " ".join([c for c in chunks if c]) or ""
    if not full_text:
        # nothing to speak
        try: FR.set_led("off")
        except Exception: pass
        try: FR.gaze_yield()
        except Exception: pass
        try:
            ASR.reset()
            ASR.pause(False)
        except Exception:
            pass
        return

    # Fire TTS builder and wait only until first chunk is produced
    try:
        TTS.tts_start(full_text, make_wav=False, blocking=False)
    except Exception as e:
        log.exception(f"TTS.tts_start failed: {e}")

    # Wait for the first mirrored 20ms block; timeout keeps system from stalling
    try:
        TTS.vap_seen_event().wait()
    except Exception:
        pass
    first_audio_ms = now_ms()

    # ---- Now send text to Furhat (device playback)
    try:
        FR.say_text_async(full_text)
    except Exception as e:
        log.exception(f"FR.say_text_async failed: {e}")
        # Recovery: resume ASR and exit
        try:
            FR.set_led("off")
            FR.gaze_yield()
        except Exception:
            pass
        try:
            ASR.reset()
            ASR.pause(False)
        except Exception:
            pass
        return

    # Anchor robot start immediately AFTER say_text_async(...)
    ROBOT_START_MS = now_ms()

    # Latencies / FTO
    tts_latency_ms = max(0, first_audio_ms - tts_call_start_ms)

    fto_raw = None
    if LAST_USER_STOP_MS is not None:
        fto_raw = ROBOT_START_MS - LAST_USER_STOP_MS
    else:
        fto_raw = None

    overlap = (fto_raw is not None and fto_raw < 0)
    fto_for_csv = None if (fto_raw is None or overlap) else fto_raw

    _csv_log(
        event_type="robot_start",
        fto_ms=fto_for_csv,
        response_time_ms=fto_for_csv,   # kept same for downstream tools
        overlap_prev_turn=1 if overlap else 0,
        tts_latency_ms=tts_latency_ms,
        user_stop_ms=LAST_USER_STOP_MS,   # first silent frame (before 1s silence)
        robot_start_ms=ROBOT_START_MS    # actual robot speech start
    )

    # Wait until device finishes speaking the full_text
    try:
        FR.speaking_done_event().wait()
    except Exception:
        pass

    # End-of-utterance bookkeeping
    ROBOT_END_MS = now_ms()
    try: FR.set_led("off")
    except Exception: pass
    try: FR.gaze_yield()
    except Exception: pass
    try:
        ASR.reset()
        ASR.pause(False)
    except Exception:
        pass

# ========= Main loop (gated by 1.0 s silence) =========
def _speak_and_log_full_text(chunks: List[str]) -> None:
    _speak_baseline_chunks(chunks)
    _append_bot(" ".join(chunks))

def _close_session_politely() -> None:
    _speak_baseline_chunks([FINAL_SENTENCE])
    _append_bot(FINAL_SENTENCE)
    _csv_log(event_type="session_end", bot_text=FINAL_SENTENCE)

def run_loop() -> None:
    opener = getattr(config, "INTERVIEW_GREETING", "Hi, welcome—shall we begin?")
    _speak_baseline_chunks([opener]); _append_bot(opener)
    if elapsed_s() >= TIME_LIMIT_S:
        _close_session_politely(); return

    global LAST_USER_START_MS, LAST_USER_STOP_MS, ASR_FINAL_MS
    global SILENCE_START_MS, USER_END_STAMPED, LAST_PARTIAL_SEEN_MS

    while True:
        if elapsed_s() >= TIME_LIMIT_S:
            _close_session_politely(); break

        LAST_USER_START_MS = None
        LAST_USER_STOP_MS = None
        ASR_FINAL_MS = None
        SILENCE_START_MS = None
        USER_END_STAMPED = False
        LAST_PARTIAL_SEEN_MS = None
        final_text: Optional[str] = None

        # --- Listen until BOTH conditions are true:
        #     (A) ASR final available, AND (B) 1.0 s silence confirmed
        while True:
            _update_audio_state_on_tick()
            nowt = now_ms()

            # Pull ASR results
            if ASR.has_new_result():
                kind, text = ASR.get_result() or (None, None)
                if kind == "partial" and text:
                    LAST_PARTIAL_SEEN_MS = nowt
                    if LAST_USER_START_MS is None:
                        LAST_USER_START_MS = nowt
                    SILENCE_START_MS = None
                    LAST_USER_STOP_MS = None
                if kind == "final" and text:
                    ASR_FINAL_MS = nowt
                    # Accumulate segments: AssemblyAI emits one final per pause,
                    # so a multi-sentence answer arrives as several finals. Append
                    # them into one turn instead of keeping only the last segment.
                    final_text = (final_text + " " + text).strip() if final_text else text
                    # New speech just arrived → restart the silence window so the
                    # turn doesn't end between segments.
                    SILENCE_START_MS = None
                    LAST_USER_STOP_MS = None

            # Silence tracking (acoustic end + 1.0 s confirm)
            speaking_flag = _detect_user_speaking_now()
            if speaking_flag is None:
                speaking_flag = _speaking_from_partials(nowt)

            if speaking_flag is True:
                SILENCE_START_MS = None
                LAST_USER_STOP_MS = None

            elif speaking_flag is False:
                if SILENCE_START_MS is None:
                    SILENCE_START_MS = nowt
                    if LAST_USER_STOP_MS is None:
                        LAST_USER_STOP_MS = SILENCE_START_MS  # acoustic end
                # 2500 ms (was 1000) — 1 s ended the turn on natural mid-sentence
                # pauses and cut users off. Matches the multi-agent SILENCE setting
                # so turn-taking is identical across both study conditions.
                if (not USER_END_STAMPED) and (nowt - SILENCE_START_MS >= 2500):
                    USER_END_STAMPED = True
                    _csv_log(event_type="user_end")

            # Exit when BOTH ASR final is ready AND silence confirmed
            if final_text and USER_END_STAMPED:
                break

            time.sleep(0.02)

        # Compute user metrics from acoustic end
        user_turn_len_ms = (max(0, LAST_USER_STOP_MS - LAST_USER_START_MS)
                            if (LAST_USER_START_MS and LAST_USER_STOP_MS) else None)
        asr_final_latency_ms = (max(0, ASR_FINAL_MS - LAST_USER_STOP_MS)
                                if (ASR_FINAL_MS and LAST_USER_STOP_MS) else None)

        _append_user(final_text, user_turn_len_ms, asr_final_latency_ms)

        # LLM streaming (after 1 s confirm by design)
        ASR.pause(True); ASR.reset()
        chunks: List[str] = []
        llm_first_logged = False
        llm_anchor = ASR_FINAL_MS or now_ms()
        for chunk, is_last in LLM.llm_stream(HISTORY):
            if (not llm_first_logged) and chunk:
                _csv_log(event_type="llm_ready",
                         llm_latency_ms=max(0, now_ms() - llm_anchor))
                llm_first_logged = True
            if chunk:
                chunks.append(chunk)
            if is_last:
                break

        if not chunks:
            try: FR.set_led("off")
            except Exception: pass
            try: FR.gaze_yield()
            except Exception: pass
            ASR.reset(); ASR.pause(False)
            continue

        _speak_and_log_full_text(chunks)

# ========= Entrypoint =========
def main() -> None:
    global START_TIME_S
    try:
        START_TIME_S = time.time()
        init_system()
        run_loop()
    except KeyboardInterrupt:
        log.info("CTRL-C: exit requested")
    except Exception as e:
        log.exception(f"FATAL: {e}")
    finally:
        shutdown_system()

if __name__ == "__main__":
    main()
