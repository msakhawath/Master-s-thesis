# =============================================================
# llm_module_ollama.py â€” Drop-in replacement for llm_module.py
# Uses a LOCAL Ollama model instead of OpenAI's cloud API.
#
# HOW TO USE:
#   In baseline_main.py, change:
#       import llm_module as LLM
#   to:
#       import llm_module_ollama as LLM
#
# Set your chosen model below in OLLAMA_MODEL.
# Everything else (streaming, PreparedReply, cancellation) is identical.
# =============================================================

from __future__ import annotations
from typing import List, Dict, Tuple, Generator, Optional
import time, threading
from queue import Queue, Empty

from openai import OpenAI  # Ollama uses the same OpenAI-compatible interface

import config
from config import gpt_system_prompt, SEPARATOR

# â”€â”€â”€ Ollama settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
OLLAMA_BASE_URL = "http://127.0.0.1:11435/v1"
OLLAMA_API_KEY  = "ollama"          # Ollama doesn't need a real key

# Must match the multi-agent runtime model (multi_agent/agents.py MODEL)
# so the baseline vs multi-agent study compares architecture, not model.
OLLAMA_MODEL    = "qwen2.5:7b"       # local Ollama model, same as multi-agent

# Persistent client: create once and reuse across all interview turns.
# This avoids ~1-2 s of client-initialisation overhead per request.
_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)

# â”€â”€â”€ Separator (same logic as llm_module.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEP = (SEPARATOR or "").strip()
if not SEP:
    SEP = " <|sep|> "

# â”€â”€â”€ Baseline system prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The full config.gpt_system_prompt is ~500 tokens (rules + job description +
# company info + topic list). qwen2.5:7b must re-process that entire prompt on
# CPU every single turn, which pushed baseline response time to ~55 s.
# This lean prompt keeps the baseline a single monolithic LLM interviewer
# (no agents, no rule-based topic structure, no safety gate â€” exactly the
# control condition) while cutting prompt-processing time dramatically. It is
# also a more faithful "unstructured baseline" than the topic-laden version.
# To restore the verbose config prompt, set USE_FULL_PROMPT = False.
USE_FULL_PROMPT = False
BASELINE_SYSTEM_PROMPT = """You are a warm professional job interviewer at DataTech Berlin interviewing for a Junior Data Scientist position.

You are the SINGLE language-model agent responsible for the whole interview.

Task:
Conduct a professional Junior Data Scientist interview using the topics below.

Output rules:
- Ask exactly ONE interview question per turn.
- Output exactly ONE short sentence containing exactly ONE question.
- Keep the complete response under 30 words.
- A brief natural filler is allowed only inside that same sentence.
- End the response with |||.
- Never ask "Are you ready for the next question?"
- Never ask two questions in one turn.
- Do not use lists, headings, quotations, or meta-commentary.

Topic-management rules:
- Work through the following topics in order.
- Ask each topic at most once.
- Determine progress yourself from the conversation history.
- Only a topic previously ASKED BY THE ASSISTANT counts as covered.
- A topic merely mentioned by the candidate does NOT count as covered.
- Do not use an external router, state machine, specialist agent, or safety controller.
- When all topics have been asked, close the interview naturally without another question.

Topics:
1. education and academic background
2. programming skills and tools
3. machine learning and modelling experience
4. data wrangling and preprocessing
5. model evaluation and metrics
6. data visualization and storytelling
7. real-world data science projects
8. teamwork and communication
9. problem solving approach
10. career goals in data science
"""

# â”€â”€â”€ Token budget helpers (identical to llm_module.py) â”€â”€â”€â”€â”€â”€â”€â”€
def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 4) if text else 0

def _estimate_tokens_in_messages(msgs: List[Dict[str, str]]) -> int:
    return sum(_rough_token_count(m.get("content", "")) for m in msgs)

def _trim_history_to_budget(messages: List[Dict[str, str]], max_total_tokens: int) -> List[Dict[str, str]]:
    if _estimate_tokens_in_messages(messages) <= max_total_tokens:
        return messages
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    sys_msg  = sys_msgs[0] if sys_msgs else None
    rest     = [m for m in messages if m.get("role") != "system"]
    kept     = ([sys_msg] if sys_msg else []) + rest[-40:]
    head     = 1 if sys_msg else 0
    while len(kept) > head and _estimate_tokens_in_messages(kept) > max_total_tokens:
        kept.pop(head)
    return kept

# â”€â”€â”€ Build messages (identical to llm_module.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    sys_prompt = gpt_system_prompt if USE_FULL_PROMPT else BASELINE_SYSTEM_PROMPT
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.extend(history or [])
    msgs.append({
        "role": "system",
        "content": f"Ask exactly ONE short interview question and append the exact delimiter {SEP} ONCE at the very end.",
    })
    return msgs

# â”€â”€â”€ Helpers (identical to llm_module.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _strip_balanced_quotes(s: str) -> str:
    if not s:
        return s
    pairs = [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")]
    for lq, rq in pairs:
        if s.startswith(lq) and s.endswith(rq) and len(s) >= 2:
            return s[1:-1].strip()
    return s

def _split_on_sep(buf: str, sep: str):
    start = 0
    nsep  = len(sep)
    while True:
        i = buf.find(sep, start)
        if i == -1:
            break
        start = i + nsep
    return start

# --- Main streaming function ---
def llm_stream(
    history: List[Dict[str, str]],
    cancel_event: Optional[threading.Event] = None,
) -> Generator[Tuple[str, bool], None, None]:
    """
    Stream the assistant reply from local Ollama, split by SEP.
    Yields (chunk, is_last) â€” identical interface to llm_module.py.
    """
    model_name    = OLLAMA_MODEL
    temperature   = float(getattr(config, "OPENAI_TEMPERATURE", 0.5))
    # Caps for the LOCAL CPU model. Interview replies are 1â€“2 short sentences,
    # so a 350-token budget and 8000-token context only make qwen2.5:7b
    # re-process a huge prompt every turn on CPU. Capping these cuts per-turn
    # latency sharply without changing the baseline's single-LLM design.
    max_tokens    = min(int(getattr(config, "OPENAI_MAX_TOKENS", 350)), 100)
    prompt_budget = min(int(getattr(config, "LLM_PROMPT_BUDGET_TOKENS", 4096)), 1500)

    msgs   = build_messages(history or [])
    msgs   = _trim_history_to_budget(msgs, prompt_budget)

    buf, held = "", ""
    stream    = None

    def _emit_not_last():
        nonlocal held
        seg = held.strip()
        if seg:
            yield (_strip_balanced_quotes(seg), False)
        held = ""

    try:
        stream = _client.chat.completions.create(
            model=model_name,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"keep_alive": "30m"},  # keep model resident between turns
        )

        for ev in stream:
            if cancel_event is not None and cancel_event.is_set():
                try:
                    stream.close()
                except Exception:
                    pass
                return

            choice = (getattr(ev, "choices", None) or [None])[0]
            if not choice:
                continue
            delta = getattr(choice, "delta", None) or {}
            piece = getattr(delta, "content", None)
            if not piece:
                if getattr(choice, "finish_reason", None) in ("stop", "length", "content_filter"):
                    break
                continue

            buf      += piece
            start_idx = _split_on_sep(buf, SEP)
            if start_idx:
                segs = [s for s in (p.strip() for p in buf[:start_idx].split(SEP)) if s]
                if segs:
                    for _it in _emit_not_last():
                        yield _it
                    for s in segs[:-1]:
                        yield (_strip_balanced_quotes(s), False)
                    held = segs[-1] if segs else held
                buf = buf[start_idx:]

            if getattr(choice, "finish_reason", None) in ("stop", "length", "content_filter"):
                break

    except Exception:
        return
    finally:
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass
        stream = None

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

    yield ("", True)


# â”€â”€â”€ Cancel / PreparedReply (identical to llm_module.py) â”€â”€â”€â”€â”€â”€
def new_cancel_event():
    return threading.Event()

def cancel_stream(cancel_event: threading.Event) -> None:
    if cancel_event is not None:
        cancel_event.set()

class PreparedReply:
    """
    Runs llm_stream in a background thread and buffers results.
    Identical interface to llm_module.PreparedReply.
    """
    def __init__(self, history):
        self.cancel_event = new_cancel_event()
        self.q: Queue[Tuple[str, bool]] = Queue(maxsize=256)
        self._done = False
        self._t = threading.Thread(target=self._run, args=(history,), daemon=True)
        self._t.start()

    def _run(self, history):
        try:
            for chunk, is_last in llm_stream(history, cancel_event=self.cancel_event):
                self.q.put((chunk, is_last))
                if is_last:
                    break
        finally:
            self._done = True

    def is_ready(self) -> bool:
        return not self.q.empty()

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


def warmup() -> None:
    """Load the model into RAM before the interview starts so the first real
    turn does not pay the one-time cold-load (~tens of seconds on CPU).
    Also pins keep_alive so the model stays resident between turns."""
    try:
            _client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            extra_body={"keep_alive": "30m"},
        )
    except Exception:
        pass


