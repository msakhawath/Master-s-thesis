# llm_module.py
"""
Lean LLM wrapper (OpenAI) — streaming chunks split by SEPARATOR
================================================================
- Uses config.gpt_system_prompt and config.SEPARATOR (trimmed to SEP).
- Streams assistant reply; splits on SEP; yields (chunk, is_last).
- Supports cancellation via threading.Event (for barge-in).
- Light history trimming against a prompt budget (rough token estimate).
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Generator, Optional
import time, random, threading, logging
from queue import Queue, Empty

# Streaming failures used to be swallowed silently, which made a dead
# controller indistinguishable from a slow one. Everything now surfaces here.
_llm_log = logging.getLogger("llm_module")
if not _llm_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    _llm_log.addHandler(_h)
_llm_log.setLevel(logging.INFO)

# Last streaming error, for callers that want to report why a reply never came.
LAST_STREAM_ERROR: Optional[str] = None

# Attempt/retry record for the most recent stream open (see _open_stream_with_retry).
LAST_ATTEMPT_INFO: dict = {"attempts": 0, "retried": False, "errors": []}

# Sequential id so a console timing line can be matched to a JSONL record.
_REQUEST_SEQ = 0

# Per-PreparedReply id, so the controller's dispatch/ready timestamps can be
# matched to the exact generation they refer to instead of being paired by order.
_PREP_SEQ = 0

# Per-request timing records, appended as JSON lines for offline analysis.
# Anchored to this file's directory, not the process working directory, so the
# records always land in the project's logs/ folder however the app is launched.
from pathlib import Path as _PathAnchor
TIMING_LOG_PATH = str(_PathAnchor(__file__).resolve().parent / "logs" / "llm_timing.jsonl")

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

import config
from config import gpt_system_prompt, SEPARATOR

# Robust delimiter (trim accidental spaces/newlines)
SEP = (SEPARATOR or "").strip()
if not SEP:
    # safe default if config forgot it
    SEP = " <|sep|> "

# ============================ TOKEN BUDGET HELPERS ============================

def _rough_token_count(text: str) -> int:
    """Very rough token estimate ≈ len/4; ensures >=1 for any non-empty string."""
    return max(1, len(text) // 4) if text else 0

def _estimate_tokens_in_messages(msgs: List[Dict[str, str]]) -> int:
    return sum(_rough_token_count(m.get("content", "")) for m in msgs)

def _trim_history_to_budget(messages: List[Dict[str, str]], max_total_tokens: int) -> List[Dict[str, str]]:
    """
    Keep system + recent turns so total ≲ budget.
    Returns a new list; does not mutate input.
    """
    if _estimate_tokens_in_messages(messages) <= max_total_tokens:
        return messages

    # Keep only the first system message (dedupe) + tail of non-system
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    sys_msg = sys_msgs[0] if sys_msgs else None
    rest = [m for m in messages if m.get("role") != "system"]

    kept = ([sys_msg] if sys_msg else []) + rest[-40:]
    # If still over budget, drop from the older end of kept (after system)
    head = 1 if sys_msg else 0
    while len(kept) > head and _estimate_tokens_in_messages(kept) > max_total_tokens:
        kept.pop(head)  # remove the oldest non-system first
    return kept

# ============================== CLIENT REUSE =================================

_CLIENT_CACHE: Dict[tuple, OpenAI] = {}


def _get_client(api_key: str, timeout_s: float, max_retries: int) -> OpenAI:
    """
    Reuse one OpenAI client instead of constructing a new one per request.

    Measured cost of per-request construction in this controller: ~4.1 s between
    PreparedReply creation and the request actually being issued (create_to_request_ms
    3933-4529 ms across 14 consecutive turns). Building the client sets up an httpx
    session and TLS context, and the main loop is meanwhile polling VAP every ~50 ms,
    so the worker thread is starved for seconds.

    llm_module_ollama.py already had this fix - "old per-request client construction
    caused ~1.2-1.8 s unnecessary overhead and was removed" - but it was never applied
    to this legacy OpenAI module.

    Keyed on the settings that define the client, so changing the key or timeout in
    config still produces a new client rather than silently reusing a stale one.
    """
    key = (api_key, float(timeout_s), int(max_retries))
    client = _CLIENT_CACHE.get(key)
    if client is None:
        t0 = time.perf_counter()
        client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=max_retries)
        _CLIENT_CACHE[key] = client
        _llm_log.info("OpenAI client constructed in %.0fms (cached for reuse)",
                      (time.perf_counter() - t0) * 1000.0)
    return client


# ============================== RETRY OPEN STREAM =============================

def _open_stream_with_retry(client: OpenAI, **kwargs):
    """
    Start streaming with light retry on transient errors (network, 429/5xx).
    Returns a streaming iterator compatible with: for chunk in stream
    """
    attempts = int(getattr(config, "OPENAI_MAX_RETRIES", 2))
    delay    = 0.35
    # LAST_ATTEMPT_INFO lets the caller report whether a retry happened, which is
    # the direct test of the "22 s outlier = timeout + retry" hypothesis. Retries
    # were previously invisible: a request that timed out at 12 s and succeeded on
    # the second attempt looked identical to one slow call.
    global LAST_ATTEMPT_INFO
    LAST_ATTEMPT_INFO = {"attempts": 0, "retried": False, "errors": []}
    for i in range(attempts + 1):
        t0 = time.perf_counter()
        try:
            LAST_ATTEMPT_INFO["attempts"] = i + 1
            LAST_ATTEMPT_INFO["retried"] = i > 0
            return client.chat.completions.create(stream=True, **kwargs)
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as e:
            took = (time.perf_counter() - t0) * 1000.0
            LAST_ATTEMPT_INFO["errors"].append(
                {"attempt": i + 1, "after_ms": round(took, 1), "error": type(e).__name__}
            )
            _llm_log.warning(
                "OPENAI ATTEMPT %d FAILED after %.0fms -> %s%s",
                i + 1, took, type(e).__name__,
                "" if i >= attempts else " (retrying)",
            )
            if i >= attempts:
                raise
            time.sleep(delay * random.uniform(0.8, 1.2))
            delay *= 2.0

# ================================= BUILD MSGS =================================

def build_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Compose OpenAI messages: single system prompt + history + SEP rule reminder.
    """
    msgs: List[Dict[str, str]] = []
    if gpt_system_prompt:
        msgs.append({"role": "system", "content": gpt_system_prompt})

    # accept already-sanitized history
    msgs.extend(history or [])

    # reinforce SEP rule (kept minimal)
    msgs.append({
        "role": "system",
        "content": f"Append the exact delimiter {SEP} after EVERY sentence or question. Keep replies concise (1–2 sentences).",
    })
    return msgs

# ================================ STREAMING API ===============================

def _strip_balanced_quotes(s: str) -> str:
    """Strip one pair of balanced quotes if present (", ', “”, ‘’)."""
    if not s:
        return s
    pairs = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]
    for lq, rq in pairs:
        if s.startswith(lq) and s.endswith(rq) and len(s) >= 2:
            return s[1:-1].strip()
    return s

def _split_on_sep(buf: str, sep: str):
    """Return the index just after the last complete SEP in buf (integer)."""
    start = 0
    nsep = len(sep)
    while True:
        i = buf.find(sep, start)
        if i == -1:
            break
        start = i + nsep
    return start


def llm_stream(history: List[Dict[str, str]],
               cancel_event: Optional["threading.Event"] = None,
               prep_id: Optional[int] = None,
               created_perf: Optional[float] = None,
               ) -> Generator[Tuple[str, bool], None, None]:
    """
    Stream the assistant reply and split it by SEP, yielding (chunk, is_last).
    - If cancel_event is set, closes stream and returns immediately.
    - Exactly one tuple will have is_last=True unless canceled early.
    """
    api_key = getattr(config, "OPENAI_API_KEY", "")
    if not api_key or not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        raise RuntimeError("Missing/invalid OPENAI_API_KEY in config.py")

    model_name  = getattr(config, "OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(getattr(config, "OPENAI_TEMPERATURE", 0.7))
    top_p       = getattr(config, "OPENAI_TOP_P", None)
    max_tokens  = int(getattr(config, "OPENAI_MAX_TOKENS", 500))
    timeout_s   = float(getattr(config, "OPENAI_REQUEST_TIMEOUT", 30.0))
    max_retries = int(getattr(config, "OPENAI_MAX_RETRIES", 2))
    prompt_budget = int(getattr(config, "LLM_PROMPT_BUDGET_TOKENS", max(512, min(4096, max_tokens * 3))))

    msgs = build_messages(history or [])
    msgs = _trim_history_to_budget(msgs, prompt_budget)

    client = _get_client(api_key, timeout_s, max_retries)

    buf, held = "", ""
    stream = None

    def _emit_not_last():
        nonlocal held
        seg = held.strip()
        if seg:
            yield (_strip_balanced_quotes(seg), False)
        held = ""

    # ---- latency instrumentation (log only, no behaviour change) -------------
    # Splits the single opaque "llm_latency_ms" into its real components, so a
    # slow reply can be attributed to network/queueing, to token generation, or
    # to the model writing a long preamble before emitting the separator.
    # perf_counter, not wall clock: these are durations, and time.time() can jump
    # under an NTP correction mid-request.
    global _REQUEST_SEQ
    _REQUEST_SEQ += 1
    _req_id = _REQUEST_SEQ

    _t_dispatch = time.perf_counter()  # T2: request about to be sent
    _t_first_token = None              # T3: first content token received
    _t_first_segment = None            # T4: first complete SEP-delimited segment
    _chars_before_sep = 0
    _chunks_before_sep = 0
    _chars_after_sep = 0
    _chunks_after_sep = 0
    _prompt_chars = sum(len(m.get("content", "")) for m in msgs)
    _reported = False

    def _report(stage: str) -> None:
        nonlocal _reported
        _reported = True

        def dur(a, b):
            return None if (a is None or b is None) else round((b - a) * 1000.0, 1)

        now = time.perf_counter()
        rec = {
            "request": _req_id,
            "prep_id": prep_id,
            # Time from PreparedReply construction (what main.py records as
            # "dispatch") to the OpenAI request actually being issued. If the
            # missing ~4 s lives in thread scheduling, prompt building or GIL
            # contention rather than the network, it shows up here.
            "create_to_request_ms": (None if created_perf is None
                                     else round((_t_dispatch - created_perf) * 1000.0, 1)),
            "stage": stage,
            "ttft_ms": dur(_t_dispatch, _t_first_token),
            "to_separator_ms": dur(_t_first_token, _t_first_segment),
            "total_to_usable_ms": dur(_t_dispatch, _t_first_segment or now),
            "total_stream_ms": dur(_t_dispatch, now),
            "chars_before_sep": _chars_before_sep,
            "chunks_before_sep": _chunks_before_sep,
            "chars_after_sep": _chars_after_sep,
            "chunks_after_sep": _chunks_after_sep,
            "separator_seen": _t_first_segment is not None and stage != "NO-SEPARATOR",
            "prompt_chars": _prompt_chars,
            "max_tokens": max_tokens,
            "model": model_name,
            "attempts": LAST_ATTEMPT_INFO.get("attempts"),
            "retried": LAST_ATTEMPT_INFO.get("retried"),
            "attempt_errors": LAST_ATTEMPT_INFO.get("errors"),
            "exception": LAST_STREAM_ERROR if stage == "EXCEPTION" else None,
        }
        _llm_log.info(
            "LLM TIMING #%d [%s] ttft=%sms to_sep=%sms total=%sms "
            "chars_before_sep=%d chars_after_sep=%d attempts=%s retried=%s",
            _req_id, stage, rec["ttft_ms"], rec["to_separator_ms"],
            rec["total_to_usable_ms"], _chars_before_sep, _chars_after_sep,
            rec["attempts"], rec["retried"],
        )
        try:
            import json as _json
            from pathlib import Path as _Path
            p = _Path(TIMING_LOG_PATH)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(_json.dumps(rec) + "\n")
        except Exception:
            pass   # diagnostics must never break a session

    try:
        kwargs = dict(model=model_name, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        if top_p is not None:
            try: kwargs["top_p"] = float(top_p)
            except Exception: pass

        _t_dispatch = time.perf_counter()
        stream = _open_stream_with_retry(client, **kwargs)

        for ev in stream:
            # Cancel?
            if cancel_event is not None and cancel_event.is_set():
                try: stream.close()
                except Exception: pass
                return

            # Parse delta content (skip tool/function deltas if any)
            choice = (getattr(ev, "choices", None) or [None])[0]
            if not choice:
                continue
            delta = getattr(choice, "delta", None) or {}
            piece = getattr(delta, "content", None)
            if not piece:
                # if tool_calls/role changes appear, ignore them
                if getattr(choice, "finish_reason", None) in ("stop", "length", "content_filter"):
                    break
                continue

            if _t_first_token is None:
                _t_first_token = time.perf_counter()   # T3: true time-to-first-token

            # Track generation before vs after the first usable segment. A large
            # chars_after_sep means the model kept writing past the separator, so
            # max_tokens / prompt verbosity is in play.
            if _t_first_segment is None:
                _chars_before_sep += len(piece)
                _chunks_before_sep += 1
            else:
                _chars_after_sep += len(piece)
                _chunks_after_sep += 1

            # Accumulate and split by SEP robustly
            buf += piece
            start_idx = _split_on_sep(buf, SEP)
            if start_idx:
                if _t_first_segment is None:
                    _t_first_segment = time.perf_counter()   # T4: first usable segment
                    _report("first-segment")
                # segments up to last SEP already yielded above via held+emit
                segs = [s for s in (p.strip() for p in buf[:start_idx].split(SEP)) if s]
                if segs:
                    for _it in _emit_not_last():
                        yield _it
                    # keep only the last as held; push others as not-last
                    for s in segs[:-1]:
                        yield (_strip_balanced_quotes(s), False)
                    held = segs[-1] if segs else held
                buf = buf[start_idx:]

            # termination signal
            if getattr(choice, "finish_reason", None) in ("stop", "length", "content_filter"):
                break

    except Exception as e:
        # End cleanly on any error (no dummy text) — but never silently.
        # A swallowed exception here leaves the controller waiting forever
        # for a reply that will never arrive.
        global LAST_STREAM_ERROR
        LAST_STREAM_ERROR = f"{type(e).__name__}: {e}"
        _llm_log.error(f"LLM STREAM FAILED -> model={model_name} {LAST_STREAM_ERROR}")
        _report("EXCEPTION")
        return
    finally:
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass
        stream = None

    # If the model never emitted the separator, nothing could be spoken until
    # generation finished completely - that alone can account for several
    # seconds of apparent "LLM latency".
    if _t_first_segment is None:
        _t_first_segment = time.perf_counter()
        _report("NO-SEPARATOR")
    elif not _reported:
        _report("stream-end")

    # Finalize: ensure buffer is flushed like model had ended with SEP
    if buf and not buf.endswith(SEP):
        buf += SEP

    if buf.strip():
        for _it in _emit_not_last():
            yield _it
        parts = [p.strip() for p in buf.split(SEP) if p.strip()]
        if parts:
            for p in parts[:-1]:
                yield (_strip_balanced_quotes(p), False)
            yield (_strip_balanced_quotes(parts[-1]), True)
            return

    last = held.strip()
    if last:
        yield (_strip_balanced_quotes(last), True)
        return

    yield ("", True)  # defensive empty last

# === Controller-facing helpers (prepare/cancel) ===

def new_cancel_event():
    return threading.Event()

def cancel_stream(cancel_event: "threading.Event") -> None:
    if cancel_event is not None:
        cancel_event.set()

class PreparedReply:
    """
    Lightweight prepared-reply buffer for the controller.
    - Spawns a background thread that runs llm_stream and buffers (chunk, is_last).
    - .is_ready() becomes True as soon as the first chunk is available.
    - .get_chunk(timeout) pops the next chunk; returns None if not ready.
    - .close() cancels the underlying stream (useful on barge-in).
    """
    def __init__(self, history):
        global _PREP_SEQ
        _PREP_SEQ += 1
        self.prep_id = _PREP_SEQ
        self.created_perf = time.perf_counter()
        self.cancel_event = new_cancel_event()
        self.q: Queue[Tuple[str, bool]] = Queue(maxsize=256)
        self._done = False
        self._produced = False
        self.error: Optional[str] = None
        self._t = threading.Thread(target=self._run, args=(history,), daemon=True)
        self._t.start()

    def _run(self, history):
        try:
            for chunk, is_last in llm_stream(history, cancel_event=self.cancel_event,
                                             prep_id=self.prep_id,
                                             created_perf=self.created_perf):
                if chunk:
                    self._produced = True
                self.q.put((chunk, is_last))
                if is_last:
                    break
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            _llm_log.error(f"PREPARED REPLY THREAD FAILED -> {self.error}")
        finally:
            if self.error is None and not self._produced:
                self.error = LAST_STREAM_ERROR or "stream produced no text"
            self._done = True

    def is_ready(self) -> bool:
        return not self.q.empty()

    def is_done(self) -> bool:
        """True once the worker thread has finished, successfully or not."""
        return self._done

    def failed(self) -> bool:
        """
        True when the reply is finished but never produced any usable text.

        The controller uses this to speak a fallback line immediately instead of
        waiting out the stall watchdog in silence.
        """
        return self._done and (not self._produced) and self.q.empty()

    def get_chunk(self, timeout: float = 0.05) -> Optional[Tuple[str, bool]]:
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        try:
            self.cancel_event.set()
        except Exception:
            pass

def llm_prepare(history) -> PreparedReply:
    return PreparedReply(history)
