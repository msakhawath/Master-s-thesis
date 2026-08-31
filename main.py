# ============================================================
# controller.py — Event-driven turn-taking controller (Upgraded for new logger)
# One-shot speaking per turn + VAP-driven gaze
# - Backchannel logging
# - Barge-in success / false-cutin handling
# - ASR / LLM / TTS latency logging
# - Strict 10-minute hard stop (between turns)
#
# METRIC ANCHORS (aligned to baseline):
#   • Acoustic END = first silent frame from ASR/VAD (LAST_USER_STOP_MS)
#   • ASR latency  = ASR_FINAL_MS − LAST_USER_STOP_MS
#   • LLM latency  = first-token − ASR_FINAL_MS
#   • Robot start  = audio onset (SPEECH_START_MS) via VAP mirrored TTS
#   • TTS latency  = SPEECH_START_MS − TTS_CALL_START_MS
#   • FTO (gap)    = SPEECH_START_MS − LAST_USER_STOP_MS
#
# DECISION (“user_end” permission per paper):
#   • VAP p_now/p_future favor robot ≥ 0.5s (MIN_GAP_TIME=500ms), or
#   • TurnGPT timeout ladder (e.g., 0.5 / 1.0 / 2.0 / 3.0 s)
#   (This decision timestamp is NOT used for metrics; only gating.)
# ============================================================

from __future__ import annotations

import time
import logging
from typing import List, Dict, Optional

# ---- Project modules
import config
import asr_module as ASR
import vap_module as VAP
import turngpt_module as TG
import similarity as SIM
import llm_module as LLM
import tts_module as TTS
import furhat_module as FR
import logger

# ---------- Logger ----------
log = logging.getLogger("controller")
log.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(_h)

# ---------- Small utilities ----------
def now_ms() -> int:
    return int(time.time() * 1000)

def _cfg(name: str, default):
    return getattr(config, name, default)

# ============================================================
# Global conversation & timing state
# ============================================================
HISTORY: List[Dict[str, str]] = []
LAST_PARTIAL: str = ""
LAST_ASR_TIME_MS: int = 0

# --- Acoustic (ASR/VAD) anchors — baseline-style
LAST_USER_START_MS: Optional[int] = None
SILENCE_START_MS: Optional[int] = None
LAST_USER_STOP_MS: Optional[int] = None       # acoustic end (first silent frame)
LAST_PARTIAL_SEEN_MS: Optional[int] = None    # for fallback “fresh partials”
ASR_FINAL_MS: Optional[int] = None            # log when ASR final arrives

# --- VAP trackers (for the *decision* logic, backchannels, barge-in)
PREV_USER_SPEAKING_VAP: bool = False

# LLM prep state
LAST_PREP_ASR: str = ""
PREP: Optional[LLM.PreparedReply] = None
PREP_STARTED_MS: int = 0
PREP_PREROLL: List[str] = []
PREP_READY_REPORTED: bool = False  # to log llm_ready once per prep

# Speaking / yield / barge-in
ROBOT_TALKING: bool = False
AWAITING_USER: bool = False
SPEECH_START_MS: int = 0           # audio onset (VAP mirror)
TTS_CALL_START_MS: int = 0
NEXT_YIELD_ALLOWED_AT: int = 0
BI_HOLD_START_MS: int = 0

# Gaze pulse state
LAST_GAZE: Optional[str] = None  # "hold" | "yield" | None
HOLD_UNTIL_MS: int = 0
NEXT_HOLD_ALLOWED_AT: int = 0
LAST_RISKY: bool = False

# Bookkeeping
LAST_BARGE_IN_AT_MS: int = 0
BARGE_IN_TRIGGERED: bool = False
LAST_USER_FINAL_AT_MS: int = 0
USER_TURN_START_MS: int = 0
USER_OVERLAP_START_MS: int = 0  # for backchannel detection

# --- Logical user turn (AssemblyAI splits one spoken answer into several finals)
# USER_TURN_OPEN stays True from the first final of an answer until the robot
# replies, so every fragment is merged into ONE history entry instead of
# becoming a separate user turn in the prompt.
USER_TURN_OPEN: bool = False
PENDING_USER_TEXT: str = ""
FALLBACK_SPOKEN_FOR_TURN: bool = False

# --- Controller-loop instrumentation (log only) -----------------------------
# Two rounds of reasoning-from-code produced wrong predictions about where the
# per-turn seconds go, so the loop now measures itself the way llm_module does.
_TICK_DURATIONS: List[int] = []
_TURN_TRACE: Dict[str, int] = {}
_LAST_BLOCK_REASON: str = ""
_LAST_STALL_WARN_MS: int = 0
CONTROLLER_TRACE_PATH = str(
    __import__("pathlib").Path(__file__).resolve().parent / "logs" / "controller_trace.jsonl"
)

# Session logger
SESSION_LOGGER: Optional[logger.SessionLogger] = None

# Time control
start_time: float = 0.0
TIME_LIMIT_S: float = 600.0  # 10 minutes
FINAL_SENTENCE: str = getattr(config, "FINAL_SENTENCE",
                              "Thank you for your time. This concludes our interview.")

# ============================================================
# Logging helpers
# ============================================================
def _log_robot_start(bot_text: str,fto_ms: Optional[int],overlap_prev_turn: int,tts_latency_ms: Optional[int], user_stop_speech: Optional[int],robot_start_speech: Optional[int],):
    if SESSION_LOGGER:
        SESSION_LOGGER.log_robot_start(
            bot_text=bot_text,
            fto_ms=fto_ms,
            overlap_prev_turn=overlap_prev_turn,
            response_time_ms=fto_ms,
            tts_latency_ms=tts_latency_ms,
            user_stop_speech=user_stop_speech,
            robot_start_speech=robot_start_speech,
        )

def _log_robot_end(robot_turn_len_ms: Optional[int]):
    if SESSION_LOGGER:
        SESSION_LOGGER.log_robot_end(robot_turn_len_ms)

def _log_user_final(user_text: str, user_turn_len_ms: Optional[int], asr_final_latency_ms: Optional[int]):
    if SESSION_LOGGER:
        SESSION_LOGGER.log_user_final(
            user_text=user_text,
            user_turn_len_ms=user_turn_len_ms,
            asr_final_latency_ms=asr_final_latency_ms,
        )

def _log_barge_in_cut():
    if SESSION_LOGGER:
        SESSION_LOGGER.log_barge_in_cut()

def _log_barge_in_success():
    if SESSION_LOGGER:
        SESSION_LOGGER.log_barge_in_success()

def _log_false_cutin():
    if SESSION_LOGGER:
        SESSION_LOGGER.log_false_cutin()

def _log_backchannel_overlap():
    if SESSION_LOGGER:
        SESSION_LOGGER.log_backchannel_overlap()

def _log_llm_ready(latency_ms: Optional[int]):
    if SESSION_LOGGER:
        SESSION_LOGGER.log_llm_ready(latency_ms)

# ============================================================
# Gaze control
# ============================================================
def _set_gaze(state: str) -> None:
    global LAST_GAZE
    s = (state or "").strip().lower()
    if s not in ("hold", "yield"):
        return
    if LAST_GAZE == s:
        return
    try:
        if s == "hold":
            FR.gaze_hold()
        else:
            FR.gaze_yield()
        LAST_GAZE = s
    except Exception:
        pass  # never crash on gaze errors

# ============================================================
# History helpers
# ============================================================
def _append_user(text: str) -> None:
    text = (text or "").strip()
    HISTORY.append({"role": "user", "content": text})
    log.info(f"[USER] {text}")

def _append_or_merge_user(text: str) -> str:
    """
    Add an ASR final to the CURRENT logical user turn.

    AssemblyAI regularly splits one spoken answer across several finals
    ("...in the code i see" / "that"). Treating each as its own turn polluted the
    prompt and let unanswered fragments pile up while the backend was stuck.
    Fragments are merged until the robot actually replies.

    Returns the full merged text of the current logical turn.
    """
    global USER_TURN_OPEN, PENDING_USER_TEXT

    text = (text or "").strip()
    if not text:
        return PENDING_USER_TEXT

    if USER_TURN_OPEN and HISTORY and HISTORY[-1].get("role") == "user":
        PENDING_USER_TEXT = f"{PENDING_USER_TEXT} {text}".strip()
        HISTORY[-1]["content"] = PENDING_USER_TEXT
        log.info(f"[USER +] {text}   (merged turn: {PENDING_USER_TEXT})")
    else:
        PENDING_USER_TEXT = text
        USER_TURN_OPEN = True
        # A new logical turn starts here, so the timing trace starts clean.
        # (Not reset in _close_user_turn: that runs inside _start_robot_speech,
        # before the trace for the turn just completed has been emitted.)
        global _LAST_BLOCK_REASON
        _TURN_TRACE.clear()
        _LAST_BLOCK_REASON = ""
        HISTORY.append({"role": "user", "content": PENDING_USER_TEXT})
        log.info(f"[USER] {PENDING_USER_TEXT}")

    return PENDING_USER_TEXT

# Strong closing markers. Deliberately NOT generic gratitude: the model uses
# "Thanks for sharing," and "I appreciate your honesty," as mid-interview fillers,
# and those must never end a session.
_CLOSING_MARKERS = (
    "concludes our interview",
    "concludes the interview",
    "participating in the interview",
    "thank you for participating",
    "appreciate your time today",
    "thank you for your time",
    "we will be in touch",
    "we'll be in touch",
    "best of luck",
    "have a nice day",
    "goodbye",
)

# Forward-looking wording. If any of these appear, the model is still driving the
# interview even if it also said something that reads like thanks, e.g.
# "Thank you for your time today. Let's move on to the next topic."
_CONTINUING_MARKERS = (
    "let's move", "lets move", "moving on", "next topic", "next question",
    "tell me", "describe", "walk me through", "explain", "let's discuss",
    "lets discuss", "let's talk", "lets talk", "shall we continue",
)


def _emit_controller_trace(bot_text: str) -> None:
    """
    One record per spoken turn: where the wall-clock actually went.

        acoustic stop -> turn settled -> prep dispatched -> prep ready -> speech

    Plus tick-duration stats, because the loop can only evaluate the settle
    condition as often as it ticks.
    """
    global _TICK_DURATIONS

    t = _TURN_TRACE
    stop = t.get("acoustic_stop_ms")

    def delta(a, b):
        return None if (a is None or b is None) else int(b - a)

    ticks = sorted(_TICK_DURATIONS)
    rec = {
        "hold_wait_ms": delta(stop, t.get("settled_ms")),
        "settle_to_dispatch_ms": delta(t.get("settled_ms"), t.get("dispatch_ms")),
        "dispatch_to_ready_ms": delta(t.get("dispatch_ms"), t.get("ready_ms")),
        "ready_to_speech_ms": delta(t.get("ready_ms"), t.get("speech_ms")),
        "total_stop_to_speech_ms": delta(stop, t.get("speech_ms")),
        "ticks": len(ticks),
        "tick_p50_ms": ticks[len(ticks) // 2] if ticks else None,
        "tick_p95_ms": ticks[min(len(ticks) - 1, int(0.95 * len(ticks)))] if ticks else None,
        "tick_max_ms": ticks[-1] if ticks else None,
        "dispatches": t.get("dispatches", 0),
        "prep_id": t.get("prep_id"),
        "last_block_reason": _LAST_BLOCK_REASON,
        "bot_text": (bot_text or "")[:80],
    }
    log.info(
        "CONTROLLER TIMING hold=%sms settle->dispatch=%sms dispatch->ready=%sms "
        "ready->speech=%sms TOTAL=%sms | ticks n=%d p50=%sms p95=%sms max=%sms | last_block=%s",
        rec["hold_wait_ms"], rec["settle_to_dispatch_ms"], rec["dispatch_to_ready_ms"],
        rec["ready_to_speech_ms"], rec["total_stop_to_speech_ms"],
        rec["ticks"], rec["tick_p50_ms"], rec["tick_p95_ms"], rec["tick_max_ms"],
        rec["last_block_reason"],
    )
    try:
        import json
        from pathlib import Path
        p = Path(CONTROLLER_TRACE_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass   # diagnostics must never break a session

    _TICK_DURATIONS = []


def _looks_like_closing(text: str) -> bool:
    """
    True when the model has ITSELF produced an interview closing.

    This does not decide when to close and adds no exit-intent rules - GPT still
    has to choose to close on its own, which is exactly the Condition B behaviour
    under study. It only lets the harness notice that the interview has ended, so
    the runtime stops instead of reopening a finished session and regenerating the
    same goodbye until Ctrl-C.

    Conservative on purpose: a closing must carry a strong marker AND ask nothing,
    because a false positive would cut a participant's session short.
    """
    import re
    body = re.sub(r"\|+", " ", (text or "")).strip()
    if not body:
        return False
    if "?" in body:                      # still asking -> not a closing
        return False
    low = re.sub(r"\s+", " ", body.lower())
    if any(marker in low for marker in _CONTINUING_MARKERS):
        return False                     # still driving the interview forward
    return any(marker in low for marker in _CLOSING_MARKERS)

def _close_user_turn() -> None:
    """Called when the robot speaks: the next final starts a fresh user turn."""
    global USER_TURN_OPEN, PENDING_USER_TEXT, FALLBACK_SPOKEN_FOR_TURN
    USER_TURN_OPEN = False
    PENDING_USER_TEXT = ""
    FALLBACK_SPOKEN_FOR_TURN = False

def _append_bot(text: str) -> None:
    text = (text or "").strip()
    HISTORY.append({"role": "assistant", "content": text})
    log.info(f"[BOT ] {text}")

# ============================================================
# ASR/VAD acoustic end detection (baseline-style)
# ============================================================
def _detect_user_speaking_now() -> Optional[bool]:
    """
    Try ASR-backed speaking flags first; fallback to VAD level; finally to fresh partials.
    Returns True/False if known, else None.
    """
    # 1) Direct speaking flag
    for fn_name in ("user_speaking", "user_is_speaking", "is_user_speaking", "vad_user_speaking"):
        fn = getattr(ASR, fn_name, None)
        if callable(fn):
            try:
                val = fn()
                if isinstance(val, bool):
                    return val
            except Exception:
                pass

    # 2) VAD energy/level
    for fn_name in ("get_vad_level", "vad_level", "get_energy_level"):
        fn = getattr(ASR, fn_name, None)
        if callable(fn):
            try:
                level = fn()
                if level is not None:
                    thr = float(getattr(config, "VAD_LEVEL_THRESHOLD", 0.5))
                    return bool(level > thr)
            except Exception:
                pass

    # 3) Fresh partial timestamp as a proxy
    for fn_name in ("last_partial_time_ms",):
        fn = getattr(ASR, fn_name, None)
        if callable(fn):
            try:
                lp = fn()
                if isinstance(lp, int):
                    fresh_ms = int(getattr(config, "PARTIAL_FRESH_MS", 180))
                    return (now_ms() - lp) <= fresh_ms
            except Exception:
                pass

    return None

def _speaking_from_partials(nowt: int) -> Optional[bool]:
    if LAST_PARTIAL_SEEN_MS is None:
        return None
    return (nowt - LAST_PARTIAL_SEEN_MS) <= int(getattr(config, "PARTIAL_FRESH_MS", 180))

def _update_audio_state_on_tick() -> None:
    """
    Updates:
      - LAST_USER_START_MS at the first detected speech in a turn
      - SILENCE_START_MS at the first frame of silence
      - LAST_USER_STOP_MS at the first silent frame (acoustic end)
    """
    global LAST_USER_START_MS, SILENCE_START_MS, LAST_USER_STOP_MS
    speaking_now = _detect_user_speaking_now()
    nowt = now_ms()

    # Fallback to partial freshness if speaking_now unknown
    if speaking_now is None:
        speaking_now = _speaking_from_partials(nowt)

    # If speaking is unknown, do nothing this tick
    if speaking_now is None:
        return

    if speaking_now is True:
        # reset silence & acoustic end while speech is ongoing
        SILENCE_START_MS = None
        LAST_USER_STOP_MS = None
        if LAST_USER_START_MS is None:
            LAST_USER_START_MS = nowt

    else:  # speaking_now is False
        if SILENCE_START_MS is None:
            SILENCE_START_MS = nowt
            if LAST_USER_STOP_MS is None:
                LAST_USER_STOP_MS = SILENCE_START_MS  # first silent frame (acoustic end)

# ============================================================
# Lifecycle
# ============================================================
def init_system() -> None:
    global LAST_ASR_TIME_MS, SESSION_LOGGER

    if not FR.ping():
        raise RuntimeError(
            f"Furhat Remote API not reachable at "
            f"http://{config.FURHAT_HOST}:{config.FURHAT_PORT}/furhat"
        )

    ASR.start_asr_stream()
    ASR.pause(False)
    SESSION_LOGGER = logger.SessionLogger(run_mode="turntaking")

    try:
        TTS.warm_up_tts(); log.info("WARMUP: TTS ok")
    except Exception as e:
        log.warning(f"WARMUP: TTS failed: {e}")
    try:
        VAP.warm_up(seconds=float(_cfg("VAP_WARMUP_S", 4.0))); log.info("WARMUP: VAP ok")
    except Exception as e:
        log.warning(f"WARMUP: VAP failed: {e}")
    try:
        TG.warm_up_turngpt(); log.info("WARMUP: TurnGPT ok")
    except Exception as e:
        log.warning(f"WARMUP: TurnGPT failed: {e}")
    try:
        SIM.warm_up_similarity(); log.info("WARMUP: Similarity ok")
    except Exception as e:
        log.warning(f"WARMUP: Similarity failed: {e}")

    now = now_ms()
    LAST_ASR_TIME_MS = now
    # seed acoustic vars to "now" so early metrics don't go negative
    global LAST_USER_START_MS, SILENCE_START_MS, LAST_USER_STOP_MS
    LAST_USER_START_MS = now
    SILENCE_START_MS = None
    LAST_USER_STOP_MS = now

    _set_gaze("yield")

def shutdown_system() -> None:
    try: FR.stop_text(barge_in=True)
    except Exception: pass
    try: TTS.stop_tts()
    except Exception: pass
    try: ASR.stop_asr_stream()
    except Exception: pass

# ============================================================
# Prepare control (LLM prefetch)
# ============================================================
def _start_prepare(reason: str, seed_asr: str) -> None:
    """(Re)start streaming LLM prep; keep CURRENT_TIME synced for the prompt."""
    global PREP, PREP_STARTED_MS, LAST_PREP_ASR, PREP_PREROLL, PREP_READY_REPORTED
    nowt = now_ms()

    # Cancel if stale or stalled
    if PREP is not None:
        stalled = ((nowt - PREP_STARTED_MS) > int(_cfg("PREP_STALL_MS", 4000))) and (not _prep_ready(peek_only=True))
        stale = (reason not in ("trigger", "refresh-partial", "watchdog")) and (PREP_STARTED_MS < LAST_ASR_TIME_MS)
        if not (stalled or stale):
            log.debug(f"PREP: keep ({reason})")
            return
        _cancel_prepare("stalled" if stalled else "stale")

    # Build msgs with up-to-date interview time
    elapsed_time = max(0.0, time.time() - start_time)
    config.CURRENT_TIME = elapsed_time

    seed = (seed_asr or "").strip()

    if reason in ("trigger", "refresh-partial", "watchdog") and seed:
        prep_history = list(HISTORY) + [
            {"role": "user", "content": seed}
        ]
    else:
        prep_history = list(HISTORY)

    PREP = LLM.llm_prepare(prep_history)

    PREP_STARTED_MS = nowt
    PREP_READY_REPORTED = False
    LAST_PREP_ASR = seed_asr or ""
    PREP_PREROLL = []
    log.debug(f"PREP: start ({reason})")

def _cancel_prepare(reason: str) -> None:
    global PREP, PREP_PREROLL, PREP_READY_REPORTED
    if PREP is None:
        return
    try:
        LLM.cancel_stream(PREP.cancel_event)
    except Exception:
        pass
    PREP = None
    PREP_PREROLL = []
    PREP_READY_REPORTED = False
    log.debug(f"PREP: cancel ({reason})")

def _prep_ready(peek_only: bool = False) -> bool:
    """Ready when first token exists; non-blocking drain. Also logs llm_ready latency once."""
    global PREP_PREROLL, PREP_READY_REPORTED, LAST_USER_FINAL_AT_MS
    if PREP is None:
        return False
    got_any = False
    while True:
        try:
            item = PREP.get_chunk(timeout=0.0)
        except Exception:
            item = None
        if item is None:
            break
        piece, is_last = item
        if piece:
            PREP_PREROLL.append(piece)
            got_any = True
            if not PREP_READY_REPORTED:
                # Measure from ASR final (baseline-aligned)
                llm_latency = now_ms() - LAST_USER_FINAL_AT_MS if LAST_USER_FINAL_AT_MS else now_ms() - PREP_STARTED_MS
                _log_llm_ready(llm_latency)
                PREP_READY_REPORTED = True
        if is_last or (peek_only and got_any):
            break

    # AMENDMENT 2026-08-20 (a) - readiness must survive a shut gate.
    #
    # PreparedReply.is_ready() is literally `not self.q.empty()`, and the drain
    # above MOVES the chunks out of that queue into PREP_PREROLL. So once the
    # reply had been drained, this function reported NOT ready even though the
    # complete text was buffered. Consequences, both observed live:
    #   - _should_yield() takes prep_ready as an argument and ends with
    #     `prep_ready and (...)`, so the robot could never speak that reply;
    #   - the PREP watchdog tests `(not _prep_ready(peek_only=True))`, so at
    #     PREP_STALL_MS it declared a completed reply "stalled", cancelled it
    #     (wiping PREP_PREROLL) and regenerated from scratch.
    # A gate shut for a single tick therefore cost 15 s + regeneration + TTS.
    # Measured on 7 of 43 turns across two pilots: ready->speech 15578-21254 ms,
    # every one containing a duplicate generation exactly PREP_STALL_MS later.
    #
    # Buffered text now counts as ready. Safe against double-speech: the speak
    # path calls _collect_prepared_text() (which empties PREP_PREROLL) and then
    # _cancel_prepare("consumed") (which sets PREP = None), after which the
    # `PREP is None` guard at the top of this function returns False.
    if got_any or PREP_PREROLL:
        return True
    try:
        ready = bool(PREP.is_ready())
        if ready and not PREP_READY_REPORTED:
            llm_latency = now_ms() - LAST_USER_FINAL_AT_MS if LAST_USER_FINAL_AT_MS else now_ms() - PREP_STARTED_MS
            _log_llm_ready(llm_latency)
            PREP_READY_REPORTED = True
        return ready
    except Exception:
        return False

def _collect_prepared_text(max_wait_ms: int = 2500) -> str:
    """
    Collect the COMPLETE prepared reply, not just whatever has arrived so far.

    The old version polled with timeout=0.0 and stopped at the first momentarily
    empty queue. Because llm_stream yields one segment per '|||', a model that
    emitted the delimiter after its opening filler produced a truncated turn -
    observed live at 12:20:23, where the robot said only "Alright,".

    Now waits for the final chunk, the worker finishing, or a bounded deadline.
    In the normal case the model places '|||' at the very end, so the last chunk
    is already queued and this returns immediately.
    """
    global PREP_PREROLL
    if PREP is None:
        return ""

    parts: List[str] = []
    if PREP_PREROLL:
        parts.extend(PREP_PREROLL)
        PREP_PREROLL = []

    deadline = now_ms() + int(max_wait_ms)
    saw_last = False

    while now_ms() < deadline:
        try:
            item = PREP.get_chunk(timeout=0.05)
        except Exception:
            item = None

        if item is None:
            # Nothing queued right now. Only stop early if the worker has
            # finished - otherwise the reply is still being generated.
            try:
                if PREP.is_done():
                    break
            except AttributeError:
                break
            continue

        piece, is_last = item
        if piece:
            parts.append(piece)
        if is_last:
            saw_last = True
            break

    if not saw_last:
        log.debug("PREP: collected without a final chunk (deadline or worker end)")

    return " ".join(parts).strip()

# ============================================================
# Yield decision (paper: VAP gate ≥ 0.5s or TurnGPT timeout ladder)
# ============================================================
def _map_tg_timeout(p: float) -> int:
    ladder = list(_cfg("TURNGPT_YIELD_TIMEOUTS",
                       [(0.30, 500), (0.20, 1000), (0.10, 2000), (0.00, 3000)]))
    for thr, ms in sorted(ladder, key=lambda x: -x[0]):
        if p >= float(thr):
            return int(ms)
    return int(ladder[-1][1])

def _should_yield(pn_u: float, pf_u: float, pn_r: float, pf_r: float,
                  gap_ms: int, tg_final: float, prep_ready: bool,
                  user_speaking_vap: bool) -> bool:
    """
    user_end (permission) when:
      - VAP favors robot (p_now_robot & p_future_robot high) AND gap_ms ≥ MIN_GAP_TIME (e.g., 500 ms), OR
      - TurnGPT timeout ladder has elapsed.
    """
    if user_speaking_vap:
        return False
    if now_ms() < NEXT_YIELD_ALLOWED_AT:
        return False

    # Minimum silence hold. TurnGPT is saturated (~0.89), so the ladder always
    # returns its top rung of 200 ms, and the robot cut in on ordinary thinking
    # pauses. This floor mirrors RQ3B_USER_EVAL_MIN_GAP_MS in the SLM arm, so the
    # two study conditions wait the same length of time before taking the floor.
    if gap_ms < int(_cfg("LEGACY_MIN_HOLD_MS", 1000)):
        return False

    user_unlikely = (pn_u <= float(_cfg("USER_LOW_NOW_FOR_YIELD", 0.43)) and
                     pf_u <= float(_cfg("USER_LOW_FUT_FOR_YIELD", 0.43)))
    robot_likely = (pn_r >= float(_cfg("VAP_PNOW_YIELD_THRESHOLD", 0.50)) and
                    pf_r >= float(_cfg("VAP_PFUT_YIELD_THRESHOLD", 0.50)))
    vap_gate_ok = (user_unlikely or robot_likely) and (gap_ms >= int(_cfg("MIN_GAP_TIME", 500)))
    tg_timeout_ok = (now_ms() - (LAST_USER_STOP_MS or now_ms())) >= _map_tg_timeout(tg_final)
    return bool(prep_ready and (vap_gate_ok or tg_timeout_ok))

# ============================================================
# Speaking (one-shot)
# ============================================================
def _start_robot_speech(text: str) -> None:
    global ROBOT_TALKING, AWAITING_USER, SPEECH_START_MS, TTS_CALL_START_MS, BARGE_IN_TRIGGERED

    t = (text or "").strip()
    if not t:
        return

    # pause ASR and mirror TTS audio to VAP first (to detect audio onset)
    ASR.pause(True)
    ASR.reset()
    tts_latency = None
    try:
        TTS_CALL_START_MS = now_ms()
        TTS.tts_start(t, make_wav=False, blocking=False)
         # Do not allow the OpenAI-TTS VAP mirror to hang the Furhat session.
        if not TTS.vap_seen_event().wait(timeout=5.0):
            log.warning("TTS mirror start timeout; continuing with Furhat speech.")

    except Exception:
        pass

    # speak on robot device
    try:
        FR.say_text_async(t)
        SPEECH_START_MS = now_ms()
        tts_latency = SPEECH_START_MS - TTS_CALL_START_MS
    except Exception:
        pass

    ROBOT_TALKING = True
    AWAITING_USER = True
    BARGE_IN_TRIGGERED = False
    # The robot has replied, so the next ASR final begins a NEW logical user turn.
    _close_user_turn()
    _append_bot(t)

    # Overlap & FTO strictly vs acoustic end
    overlap = 1 if (PREV_USER_SPEAKING_VAP or (LAST_USER_STOP_MS and SPEECH_START_MS < LAST_USER_STOP_MS)) else 0
    fto = (SPEECH_START_MS - LAST_USER_STOP_MS) if (LAST_USER_STOP_MS and not overlap) else None
    _log_robot_start(bot_text=t, fto_ms=fto, overlap_prev_turn=overlap,tts_latency_ms=tts_latency,user_stop_speech=LAST_USER_STOP_MS,robot_start_speech=SPEECH_START_MS )

# ============================================================
# Barge-in helpers
# ============================================================
def _vap_interrupt_imminent(pn_u: float, pf_u: float) -> bool:
    thr_now = float(_cfg("VAP_USER_PNOW_INTERRUPT", 0.48))
    thr_fut = float(_cfg("VAP_USER_PFUT_INTERRUPT", 0.48))
    return (pn_u >= thr_now) and (pf_u >= thr_fut)

def _handle_barge_in(pn_u: float, pf_u: float, user_speaking_vap: bool) -> bool:
    """True if we cut robot speech due to a validated user barge-in."""
    global BI_HOLD_START_MS, ROBOT_TALKING, NEXT_YIELD_ALLOWED_AT
    global HOLD_UNTIL_MS, NEXT_HOLD_ALLOWED_AT, LAST_RISKY, LAST_BARGE_IN_AT_MS, BARGE_IN_TRIGGERED
        # The legacy Furhat path speaks with Furhat TTS while VAP's robot
    # reference is generated separately by OpenAI TTS. This mismatch can
    # make Furhat's own loudspeaker leakage look like user speech.
    # Disable automatic cutting for the participant-facing legacy system.
    if not bool(_cfg("ENABLE_BARGE_IN", False)):
        BI_HOLD_START_MS = 0
        return False

    if not ROBOT_TALKING:
        BI_HOLD_START_MS = 0
        return False

    # grace window to avoid false positives right at TTS onset
    if (now_ms() - SPEECH_START_MS) < int(_cfg("BARGE_IN_GRACE_MS", 350)):
        BI_HOLD_START_MS = 0
        return False

    if not user_speaking_vap:
        BI_HOLD_START_MS = 0
        return False

    # Optional: avoid cutting in "backchannel band"
    in_back_band = (float(_cfg("BACK_PNOW_MIN", 0.33)) <= pn_u <= float(_cfg("BACK_PNOW_MAX", 0.65))) \
                   and (pf_u <= float(_cfg("BACK_PFUT_MAX", 0.35)))
    if in_back_band:
        BI_HOLD_START_MS = 0
        return False

    if not _vap_interrupt_imminent(pn_u, pf_u):
        BI_HOLD_START_MS = 0
        return False

    if BI_HOLD_START_MS == 0:
        BI_HOLD_START_MS = now_ms()
        return False

    if (now_ms() - BI_HOLD_START_MS) < int(_cfg("BARGE_IN_HOLD_MS", 300)):
        return False

    # Cut TTS
    try: FR.stop_text(barge_in=True)
    except Exception: pass
    try: TTS.stop_tts()
    except Exception: pass

    ROBOT_TALKING = False
    BARGE_IN_TRIGGERED = True
    ASR.reset(); ASR.pause(False)
    NEXT_YIELD_ALLOWED_AT = now_ms() + int(_cfg("YIELD_COOLDOWN_MS", 400))

    _set_gaze("yield")
    HOLD_UNTIL_MS = 0
    NEXT_HOLD_ALLOWED_AT = 0
    LAST_RISKY = False

    _log_barge_in_cut()
    LAST_BARGE_IN_AT_MS = now_ms()
    log.info("BARGE-IN: cut")
    return True

# ============================================================
# Main event loop
# ============================================================
def run_loop() -> None:
    global LAST_PARTIAL, LAST_ASR_TIME_MS, ASR_FINAL_MS
    global ROBOT_TALKING, AWAITING_USER, NEXT_YIELD_ALLOWED_AT
    global PREV_USER_SPEAKING_VAP, LAST_PREP_ASR
    global HOLD_UNTIL_MS, NEXT_HOLD_ALLOWED_AT, LAST_RISKY
    global LAST_USER_FINAL_AT_MS, LAST_BARGE_IN_AT_MS, USER_TURN_START_MS, USER_OVERLAP_START_MS, BARGE_IN_TRIGGERED
    global LAST_USER_START_MS, SILENCE_START_MS, LAST_USER_STOP_MS, LAST_PARTIAL_SEEN_MS
    global USER_TURN_OPEN, PENDING_USER_TEXT, FALLBACK_SPOKEN_FOR_TURN
    global _LAST_BLOCK_REASON, _LAST_STALL_WARN_MS

    # GREETING
    if not HISTORY:
        opener = _cfg("INTERVIEW_GREETING", "Hi, welcome—shall we begin?")
        _start_robot_speech(opener)
        nowt = now_ms()
        LAST_ASR_TIME_MS = nowt
        LAST_USER_START_MS = nowt
        LAST_USER_STOP_MS = nowt
        _set_gaze("yield")

    while True:
        tick_ms = now_ms()

        # ---- 10-minute cutoff (only between turns) ----
        elapsed_s = max(0.0, time.time() - start_time)
        config.CURRENT_TIME = elapsed_s  # let LLM know the running time if needed
        time_limit_reached = (elapsed_s >= TIME_LIMIT_S)

        # VAP snapshot for decision / backchannels / barge-in
        win = float(_cfg("VAP_IDLE_WIN", 0.24)) if not ROBOT_TALKING else float(_cfg("VAP_SPEAK_WIN", 0.24))
        v = VAP.get_current(window_sec=win)
        pn_u = float(v.get("p_now_user", 0.5))
        pf_u = float(v.get("p_fut_user", 0.5))
        pn_r = float(v.get("p_now_robot", 0.5))
        pf_r = float(v.get("p_fut_robot", 0.5))
        user_speaking_vap = bool(v.get("user_is_speaking", 0))

        # Time limit close-out (only when both are silent)
        if time_limit_reached and (not ROBOT_TALKING) and (not user_speaking_vap):
            _start_robot_speech(FINAL_SENTENCE)
            log.info("TIME LIMIT: final sentence delivered, exiting loop.")
            while ROBOT_TALKING and not FR.speaking_done_event().is_set():
                time.sleep(0.05)
            break

        # ------ ASR/VAD acoustic state update (every tick) ------
        _update_audio_state_on_tick()

        # -------------------- Backchannel detection --------------------
        # While robot is talking:
        if ROBOT_TALKING:
            if user_speaking_vap and USER_OVERLAP_START_MS == 0:
                USER_OVERLAP_START_MS = tick_ms
            elif (not user_speaking_vap) and USER_OVERLAP_START_MS:
                overlap_dur = tick_ms - USER_OVERLAP_START_MS
                # Short overlap, no barge-in => backchannel
                if overlap_dur < int(_cfg("BACKCHANNEL_DEADZONE_MS", 800)) and not BARGE_IN_TRIGGERED:
                    _log_backchannel_overlap()
                USER_OVERLAP_START_MS = 0
                BARGE_IN_TRIGGERED = False
        else:
            USER_OVERLAP_START_MS = 0
            BARGE_IN_TRIGGERED = False
        # ----------------------------------------------------------------

        # -------------------- Dynamic gaze --------------------
        if user_speaking_vap:
            _set_gaze("yield")
            HOLD_UNTIL_MS = 0
            LAST_RISKY = False
        else:
            if ROBOT_TALKING:
                risky = (pf_u >= pf_r)
                if risky and (not LAST_RISKY) and (tick_ms >= NEXT_HOLD_ALLOWED_AT):
                    _set_gaze("hold")
                    HOLD_UNTIL_MS = tick_ms + int(_cfg("GAZE_HOLD_PULSE_MS", 400))
                    NEXT_HOLD_ALLOWED_AT = tick_ms + int(_cfg("GAZE_HOLD_REFRACTORY_MS", 350))
                if HOLD_UNTIL_MS and (tick_ms >= HOLD_UNTIL_MS):
                    _set_gaze("yield")
                    HOLD_UNTIL_MS = 0
                LAST_RISKY = risky
            else:
                _set_gaze("yield")
                HOLD_UNTIL_MS = 0
                NEXT_HOLD_ALLOWED_AT = 0
                LAST_RISKY = False
        # ------------------------------------------------------

        # --------- Drain ALL queued ASR events this tick ---------
        # Draining only one event per tick let the queue fall behind the speaker,
        # which is what produced the sounddevice input-overflow warnings.
        _asr_drained = 0
        while ASR.has_new_result() and _asr_drained < 64:
            _asr_drained += 1
            kind, text = ASR.get_result() or (None, None)
            if kind:
                LAST_ASR_TIME_MS = tick_ms

                if kind == "partial":
                    LAST_PARTIAL = text or ""
                    LAST_PARTIAL_SEEN_MS = tick_ms
                    AWAITING_USER = False
                    if USER_TURN_START_MS == 0:
                        USER_TURN_START_MS = tick_ms
                    # DEBUG, not INFO: one line per partial made the console
                    # unreadable during a session.
                    log.debug(f"ASR PARTIAL: {LAST_PARTIAL}")

                elif kind == "final":
                    final_text = text or ""
                    prev_partial = LAST_PARTIAL
                    # Merge into the current logical turn rather than appending
                    # a new history entry per ASR fragment.
                    merged_text = _append_or_merge_user(final_text)
                    LAST_PARTIAL = ""
                    AWAITING_USER = False

                    # Compute metrics strictly vs acoustic end
                    user_len = (tick_ms - USER_TURN_START_MS) if USER_TURN_START_MS else None
                    asr_lat = (tick_ms - LAST_USER_STOP_MS) if LAST_USER_STOP_MS else None
                    _log_user_final(user_text=final_text, user_turn_len_ms=user_len, asr_final_latency_ms=asr_lat)

                    USER_TURN_START_MS = 0
                    LAST_USER_FINAL_AT_MS = tick_ms
                    ASR_FINAL_MS = tick_ms

                    # Similarity checks to decide whether to reuse PREP.
                    # Compared against the MERGED turn, since that is what the
                    # prompt now contains.
                    similar_to_prev = False
                    try:
                        thr = float(_cfg("SIMILARITY_PREPARE_THRESHOLD", 0.80))
                        if prev_partial:
                            s = SIM.compute_similarity(merged_text, prev_partial)
                            similar_to_prev = (s >= thr)
                    except Exception:
                        similar_to_prev = False

                    similar_to_seed = False
                    if (not similar_to_prev) and LAST_PREP_ASR:
                        try:
                            thr = float(_cfg("SIMILARITY_PREPARE_THRESHOLD", 0.80))
                            s = SIM.compute_similarity(merged_text, LAST_PREP_ASR)
                            similar_to_seed = (s >= thr)
                        except Exception:
                            similar_to_seed = False

                    # Generation is NOT started here any more.
                    #
                    # Starting (and restarting) a generation on every ASR fragment
                    # produced several concurrent OpenAI streams per turn - 30
                    # generations for 21 spoken turns in the pilot, nine of them
                    # discarded - and pushed median FTO to 6.0 s. cancel_stream()
                    # only takes effect when a stream next yields, so abandoned
                    # requests kept running and competed with the live one.
                    #
                    # Instead the turn-settled block below starts exactly ONE
                    # generation, once the participant has actually stopped for
                    # LEGACY_MIN_HOLD_MS. Any in-flight generation is dropped here,
                    # because this new fragment means the answer was not finished.
                    if PREP is not None and not (similar_to_prev or similar_to_seed):
                        _cancel_prepare("turn-grew")

        # --------- Turn-taking decision (VAP+TurnGPT) ---------
        # TurnGPT costs ~1.6 s per call on CPU (measured). Calling it twice on
        # every tick made one loop iteration ~3.3 s, so the settle condition could
        # only be evaluated every 3.3 s and a 1000 ms hold became a 4-5 s wait.
        # That was the entire unexplained local delay: the API itself answers in
        # ~900 ms. Both calls are now made only when they can change the outcome.
        #
        # Semantics are unchanged. _should_yield() cannot return True while
        # gap_ms < LEGACY_MIN_HOLD_MS, so a TurnGPT score computed before then
        # provably cannot affect the decision.
        has_user_turn = any(
            m.get("role") == "user"
            and (m.get("content") or "").strip()
            for m in HISTORY
        )

        # for_prepare is read ONLY by the EARLY PREPARE block below, which is off
        # for the study. Computing it unconditionally burned ~1.6 s per tick on a
        # value nothing consumed.
        tg_p_prepare = 0.0
        if bool(_cfg("ENABLE_EARLY_LLM_PREFETCH", False)):
            if has_user_turn or bool((LAST_PARTIAL or "").strip()):
                tg_p_prepare = TG.get_turngpt_score(
                    HISTORY,
                    partial=(LAST_PARTIAL or None),
                    for_prepare=True,
                )

        # for_final feeds the yield ladder, which cannot fire below the hold.
        #
        # ORDERING NOTE: this is deliberately computed BEFORE the TURN SETTLED
        # block, i.e. before the LLM request is dispatched. Scoring after dispatch
        # would overlap the ~1.5 s TurnGPT computation with the ~0.9 s API call and
        # cut roughly 1.5 s from FTO -- but that is a latency optimisation, not a
        # neutral change, and overlapping work would mask the still-unexplained
        # ~5 s dispatch->ready gap. Measure first, optimise afterwards.
        _gap_now = tick_ms - (LAST_USER_STOP_MS or tick_ms)
        tg_p_final = 0.0
        if (has_user_turn and (not ROBOT_TALKING) and (not AWAITING_USER)
                and _gap_now >= int(_cfg("LEGACY_MIN_HOLD_MS", 1000))):
            tg_p_final = TG.get_turngpt_score(
                HISTORY,
                partial=None,
                for_prepare=False,
            )


        # EARLY PREPARE: speculative generation from evolving ASR partials.
        # Disabled for the participant-facing study via ENABLE_EARLY_LLM_PREFETCH.
        # Long answers produced a stream of partials, and each one could cancel and
        # restart the in-flight OpenAI request, so no reply ever completed. The SLM
        # arm also has prefetch off, so both study arms now generate only after a
        # settled final transcript.
        if bool(_cfg("ENABLE_EARLY_LLM_PREFETCH", False)) and (not AWAITING_USER):
            tg_ok = bool(LAST_PARTIAL) and (tg_p_prepare >= float(_cfg("TURNGPT_PREPARE_THRESHOLD", 0.20)))
            quiet_ok = (tick_ms - LAST_ASR_TIME_MS) >= int(_cfg("ASR_PREPARE_TIMEOUT_MS", 200))
            if tg_ok or quiet_ok:
                current_asr = (LAST_PARTIAL or "").strip()
                if current_asr:
                    similar_seed = False
                    try:
                        if LAST_PREP_ASR:
                            thr = float(_cfg("SIMILARITY_PREPARE_THRESHOLD", 0.80))
                            s = SIM.compute_similarity(current_asr, LAST_PREP_ASR)
                            similar_seed = (s >= thr)
                    except Exception:
                        similar_seed = False

                    deb_ms = int(_cfg("PREP_REFRESH_DEBOUNCE_MS", 350))
                    min_delta = int(_cfg("PREP_MIN_DELTA_CHARS", 3))
                    len_delta_ok = (len(current_asr) - len(LAST_PREP_ASR or "")) >= min_delta
                    ready_now = _prep_ready(peek_only=True)
                    age_ok = (PREP is None) or ((tick_ms - PREP_STARTED_MS) >= deb_ms)

                    if PREP is None:
                        _start_prepare("trigger", seed_asr=current_asr)
                    elif (not similar_seed) and (not ready_now) and age_ok and len_delta_ok:
                        _cancel_prepare("early-refresh")
                        _start_prepare("trigger", seed_asr=current_asr)

        # Yield decision (user_end permission) — GAP measured vs acoustic end
        gap_ms = tick_ms - (LAST_USER_STOP_MS or tick_ms)

        # Track the most recent acoustic stop for the open turn - gap_ms is
        # measured from it, so it is the anchor for the whole timing chain.
        # Frozen once the turn settles: continuing to overwrite it after dispatch
        # produced negative hold_wait_ms whenever the participant spoke again.
        if USER_TURN_OPEN and LAST_USER_STOP_MS and "settled_ms" not in _TURN_TRACE:
            _TURN_TRACE["acoustic_stop_ms"] = LAST_USER_STOP_MS

        # ---- TURN SETTLED: start exactly one generation ----------------------
        # The participant has stopped for the full hold, so the answer is treated
        # as complete and generated from the MERGED turn text. One request per
        # turn, no speculative restarts, no overlapping streams.
        _hold_ms = int(_cfg("LEGACY_MIN_HOLD_MS", 1000))
        _settle_ok = (PREP is None and USER_TURN_OPEN and (not ROBOT_TALKING)
                      and (not user_speaking_vap)
                      and gap_ms >= _hold_ms)

        if _settle_ok:
            _TURN_TRACE.setdefault("settled_ms", tick_ms)
            _start_prepare("turn-settled", seed_asr=PENDING_USER_TEXT)
            # dispatch/ready are per-GENERATION, not per-turn: a re-dispatch after
            # the participant resumed must not be paired with the earlier ready,
            # which is what produced negative dispatch_to_ready values.
            _TURN_TRACE["dispatch_ms"] = now_ms()
            _TURN_TRACE.pop("ready_ms", None)
            _TURN_TRACE["dispatches"] = _TURN_TRACE.get("dispatches", 0) + 1
            _TURN_TRACE["prep_id"] = getattr(PREP, "prep_id", None)
            log.debug(f"PREP: turn settled after {gap_ms}ms -> generating once")

        elif USER_TURN_OPEN and (not ROBOT_TALKING):
            # Record WHICH condition is holding the turn back. Without this a
            # stall is indistinguishable from the participant still speaking.
            if PREP is not None:
                _LAST_BLOCK_REASON = "prep_already_running"
            elif user_speaking_vap:
                _LAST_BLOCK_REASON = "vap_says_user_speaking"
            elif gap_ms < _hold_ms:
                _LAST_BLOCK_REASON = f"gap_{gap_ms}ms_below_hold_{_hold_ms}ms"
            else:
                _LAST_BLOCK_REASON = "unknown"

        # ---- Stall detector --------------------------------------------------
        # A 31 s silence appeared in the 11:16 pilot with no generation attempted.
        # This says out loud what was blocking, rather than leaving it to inference.
        if (USER_TURN_OPEN and PREP is None and (not ROBOT_TALKING)
                and LAST_USER_FINAL_AT_MS
                and (tick_ms - LAST_USER_FINAL_AT_MS) > 5000
                and (tick_ms - _LAST_STALL_WARN_MS) > 5000):
            _LAST_STALL_WARN_MS = tick_ms
            log.warning(
                "CONTROLLER STALL: %dms since last user final, no generation started. "
                "blocked_by=%s gap_ms=%d vap_speaking=%s awaiting_user=%s "
                "robot_talking=%s pending_text_len=%d",
                tick_ms - LAST_USER_FINAL_AT_MS, _LAST_BLOCK_REASON, gap_ms,
                user_speaking_vap, AWAITING_USER, ROBOT_TALKING, len(PENDING_USER_TEXT or ""),
            )

        if (not ROBOT_TALKING) and (not AWAITING_USER):
            ready = _prep_ready(peek_only=True)
            # Only record readiness for the generation we actually dispatched;
            # pairing by order produced negative dispatch_to_ready values.
            if (ready and "ready_ms" not in _TURN_TRACE
                    and getattr(PREP, "prep_id", None) == _TURN_TRACE.get("prep_id")):
                _TURN_TRACE["ready_ms"] = now_ms()

            # ---- Backend failure fallback -------------------------------------
            # _should_yield() requires prep_ready, so a failed generation used to
            # leave the robot permanently silent while the watchdog restarted it
            # every few seconds. Detect a finished-but-empty reply and say
            # something, so the participant is never left in open-ended silence.
            prep_failed = False
            if (PREP is not None) and (not ready):
                try:
                    prep_failed = bool(PREP.failed())
                except AttributeError:
                    prep_failed = False

            if (prep_failed and USER_TURN_OPEN and (not FALLBACK_SPOKEN_FOR_TURN)
                    and (not user_speaking_vap) and gap_ms >= int(_cfg("MIN_GAP_TIME", 500))):
                err = getattr(PREP, "error", None) or "unknown backend failure"
                log.error(f"LLM PREP FAILED -> speaking fallback. reason={err}")
                _cancel_prepare("failed")
                FALLBACK_SPOKEN_FOR_TURN = True
                _start_robot_speech(str(_cfg("LLM_FALLBACK_TEXT",
                                             "Sorry, I did not catch that. Could you say it again?")))
                LAST_ASR_TIME_MS = tick_ms
                LAST_PARTIAL = ""

            elif _should_yield(pn_u, pf_u, pn_r, pf_r, gap_ms, tg_p_final, ready, user_speaking_vap):
                if user_speaking_vap:
                    NEXT_YIELD_ALLOWED_AT = now_ms() + int(_cfg("YIELD_COOLDOWN_MS", 400))
                else:
                    prepared = _collect_prepared_text()
                    if prepared:
                        closing = _looks_like_closing(prepared)
                        _start_robot_speech(prepared)
                        _TURN_TRACE["speech_ms"] = SPEECH_START_MS
                        _emit_controller_trace(prepared)
                        LAST_ASR_TIME_MS = tick_ms
                        LAST_PARTIAL = ""
                        _cancel_prepare("consumed")

                        if closing:
                            # The model delivered a closing. Speak it ONCE, record
                            # it, then stop. Previously the runtime reopened the
                            # finished interview, took another "thank you", and
                            # regenerated the same goodbye until Ctrl-C.
                            # This changes only the harness lifecycle - whether GPT
                            # recognises that it should close is untouched, and
                            # remains the Condition B behaviour under study.
                            while ROBOT_TALKING and not FR.speaking_done_event().is_set():
                                time.sleep(0.05)
                            _log_robot_end(robot_turn_len_ms=now_ms() - SPEECH_START_MS)
                            if SESSION_LOGGER:
                                try:
                                    SESSION_LOGGER.log_event(
                                        event_type="closing_delivered",
                                        bot_text=prepared,
                                    )
                                except Exception:
                                    pass
                            log.info("CLOSING DELIVERED by the model - ending session.")
                            break

        # Speaking monitor (one-shot; no chaining)
        if ROBOT_TALKING:
            if _handle_barge_in(pn_u, pf_u, user_speaking_vap):
                _cancel_prepare("after-barge-in")
                if LAST_PARTIAL:
                    _start_prepare("trigger", seed_asr=LAST_PARTIAL)
            else:
                # End of the robot turn (no chaining)
                if FR.speaking_done_event().is_set() or \
                   ((now_ms() - SPEECH_START_MS) >= int(_cfg("TEXTMODE_MAX_TURN_MS", 25000))):
                    try: TTS.stop_tts()
                    except Exception: pass
                    robot_len = now_ms() - SPEECH_START_MS
                    _log_robot_end(robot_turn_len_ms=robot_len)

                    ROBOT_TALKING = False
                    ASR.reset(); ASR.pause(False)
                    NEXT_YIELD_ALLOWED_AT = now_ms() + int(_cfg("YIELD_COOLDOWN_MS", 400))
                    _set_gaze("yield")
                    HOLD_UNTIL_MS = 0
                    NEXT_HOLD_ALLOWED_AT = 0
                    LAST_RISKY = False

        # PREP watchdog
        if PREP is not None:
            stalled = ((now_ms() - PREP_STARTED_MS) > int(_cfg("PREP_STALL_MS", 4000))) and (not _prep_ready(peek_only=True))
            if stalled and (not AWAITING_USER) and (not user_speaking_vap):
                _cancel_prepare("stall")
                _start_prepare("watchdog", seed_asr=(LAST_PARTIAL or ""))

        # Inactivity reprompt (no repair flag per your spec)
        #
        # AMENDMENT 2026-08-20 (b) - never reprompt over work in progress.
        #
        # This fired on elapsed-ASR-time alone. It checked neither PREP nor
        # USER_TURN_OPEN, so it could interrupt a fully generated reply and then
        # call _start_robot_speech() -> _close_user_turn(), DISCARDING the
        # participant's pending answer and orphaning the PREP that blocks the
        # next settle with prep_already_running. Caught in the act six times in
        # the 23:52 pilot, every one with user_turn_open=True and 3-114 chars of
        # real pending text destroyed; the participant answered, was asked "are
        # you there?", said "I already answered you", and the loop compounded
        # until they said "i don't want to continue you don't understand me".
        #
        # The reprompt is for a genuinely absent participant, so it must now
        # also see no live generation, no buffered reply, and no unanswered
        # turn. Silence with work pending is a controller problem, not an
        # absent participant, and speaking over it destroys data.
        if (tick_ms - LAST_ASR_TIME_MS) > int(_cfg("INACTIVITY_REPROMPT_MS", 30000)) and (not ROBOT_TALKING) and (not user_speaking_vap) and (PREP is None) and (not PREP_PREROLL) and not (USER_TURN_OPEN and (PENDING_USER_TEXT or "").strip()):
            reprompt = _cfg("INACTIVITY_REPROMPT_TEXT", "are you there ? shall we continue?")
            _start_robot_speech(reprompt)
            LAST_ASR_TIME_MS = tick_ms

        # Barge-in outcome tagging (success vs false cut-in)
        if LAST_BARGE_IN_AT_MS:
            success_win = int(_cfg("BARGE_IN_GRACE_MS", 350))  # reuse grace as window
            timeout_ms = int(_cfg("FALSE_CUTIN_TIMEOUT_MS", 1200))
            # success: user starts speaking within window after cut
            if user_speaking_vap and (now_ms() - LAST_BARGE_IN_AT_MS) <= success_win:
                _log_barge_in_success()
                LAST_BARGE_IN_AT_MS = 0
            # false cut-in: no user speech until timeout, and no user final
            elif (tick_ms - LAST_BARGE_IN_AT_MS) > timeout_ms and (LAST_USER_FINAL_AT_MS < LAST_BARGE_IN_AT_MS):
                _log_false_cutin()
                LAST_BARGE_IN_AT_MS = 0
            # user did speak (final arrived) but outside window → clear bookkeeping
            elif LAST_USER_FINAL_AT_MS >= LAST_BARGE_IN_AT_MS:
                LAST_BARGE_IN_AT_MS = 0

        # pacing
        chunk_ms = int(_cfg("AUDIO_CHUNK_MS", 50))
        spent = now_ms() - tick_ms
        # How long this iteration actually took. The settle condition can only be
        # evaluated once per tick, so a slow tick directly becomes response delay.
        _TICK_DURATIONS.append(spent)
        time.sleep(max(5, chunk_ms - spent) / 1000.0)

# ============================================================
# Entrypoint
# ============================================================
def main() -> None:
    try:
        init_system()
        global start_time
        start_time = time.time()
        run_loop()
    except KeyboardInterrupt:
        log.info("CTRL-C: exit requested")
    except Exception as e:
        log.exception(f"FATAL: {e}")
    finally:
        shutdown_system()

if __name__ == "__main__":
    main()
