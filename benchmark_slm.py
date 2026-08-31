#!/usr/bin/env python3
# =============================================================
# benchmark_slm.py — thesis-grade repeated SLM benchmark
# =============================================================
# Key methodological changes:
#   1. Cold-start/model-load measurement is SEPARATE from interview tasks.
#   2. Each task is repeated multiple times (default 10).
#   3. Cold load is repeated independently (default 3) using Ollama's native
#      load_duration metric after explicitly unloading the model.
#   4. Uses time.perf_counter() for elapsed client-side timing.
#   5. Randomizes task order within each repetition.
#   6. Uses reproducible seeds.
#   7. Measures strict separator formatting, not just "separator appears".
#   8. Adds an automatic question-count heuristic and objective rule checks.
#   9. Saves model/quantization/hardware metadata for reproducibility.
#  10. Generates a BLINDED human-rating sheet; model identity is omitted.
#
# Examples:
#   python benchmark_slm.py --profile local
#   python benchmark_slm.py --profile server --repeats 10 --cold-repeats 3
#   python benchmark_slm.py --only gemma3:1b --repeats 10
#   python benchmark_slm.py --models qwen3:8b,llama3.1:8b,gemma2:9b
# =============================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from openai import OpenAI


SEPARATOR = "|||"
WORD_LIMIT = 35
DEFAULT_SEED = 20260601
WARM_KEEP_ALIVE = "30m"


# -----------------------------------------------------------------------------
# Model sets
# -----------------------------------------------------------------------------
# Keep the original local set for direct comparison with your previous run.
MODELS_LOCAL = [
    "phi4-mini",
    "qwen3:4b",
    "gemma3:4b",
    "gemma3:1b",
    "llama3.2",
    "phi3:mini",
]

# Recommended sub-10B university-server set.
# Includes the deployed 1B model as an anchor, the previous 4B winner, and
# several ~8-9B candidates from different model families.
MODELS_SERVER = [
    "gemma3:1b",
    "gemma3:4b",
    "phi4-mini:3.8b",
    "qwen3:8b",
    "llama3.1:8b",
    "ministral-3:8b",
    "gemma2:9b",
]


# -----------------------------------------------------------------------------
# Benchmark prompt and tasks
# -----------------------------------------------------------------------------
# Important correction: the earlier benchmark globally demanded a question on
# every reply but also treated the "feedback" task as a closing-cue test. Those
# goals conflict. This prompt explicitly defines the closing exception.
SYSTEM_PROMPT = f"""You are a warm, professional job interviewer at DataTech Berlin for a Junior Data Scientist role.

Follow these rules exactly:
- For a normal interview turn, ask exactly ONE relevant question.
- If the candidate clearly signals that the interview is ending, give ONE short closing statement and ask ZERO questions.
- Keep the entire reply under {WORD_LIMIT} words.
- Keep the reply concise and natural; do not use lists, bullets, hashtags, headings, or meta-commentary.
- Put the exact separator {SEPARATOR} ONCE, at the very end of the entire reply.
- Do not output any other pipe characters.
""".strip()

TASKS = [
    {
        "id": "warmup",
        "label": "Warm-up question",
        "user_msg": "Hi, I am ready for the interview.",
        "expected_questions": 1,
        "expected_mode": "question",
    },
    {
        "id": "star_probe",
        "label": "Behavioral follow-up",
        "user_msg": "I built a machine learning model for a university project with a team of four people.",
        "expected_questions": 1,
        "expected_mode": "question",
    },
    {
        "id": "weak_answer",
        "label": "Handling a weak answer",
        "user_msg": "I don't really have experience with machine learning or model evaluation.",
        "expected_questions": 1,
        "expected_mode": "question",
    },
    {
        "id": "closing_cue",
        "label": "Recognising a closing cue",
        "user_msg": "That is all I have to say about my data science experience. Thank you.",
        "expected_questions": 0,
        "expected_mode": "closing",
    },
    {
        "id": "instruction_follow",
        "label": "Explicit topic request",
        "user_msg": "Can you ask me something about data visualization?",
        "expected_questions": 1,
        "expected_mode": "question",
    },
]

# Cold-start prompt is NOT one of the five evaluated interview tasks.
COLD_START_PROMPT = "Reply with exactly: OK"
RESOURCE_PROBE_PROMPT = "In one concise sentence, explain the difference between precision and recall."


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated benchmark for interview SLMs.")
    parser.add_argument("--profile", choices=["local", "server"], default="local")
    parser.add_argument("--only", default=None, help="Benchmark one exact model ID.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated exact Ollama model IDs; overrides --profile.",
    )
    parser.add_argument("--label", default="", help="Optional output filename label.")
    parser.add_argument("--repeats", type=int, default=10, help="Warm repetitions per task (10 recommended for the final thesis run).")
    parser.add_argument(
        "--cold-repeats",
        type=int,
        default=3,
        help="Independent cold-load repetitions per model.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout in seconds for warm streamed tasks.")
    parser.add_argument("--resource-repeats", type=int, default=3, help="Repeated native Ollama throughput probes per model.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible Ollama URL. Run the script on the university server and localhost is fine; otherwise supply the server URL.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Stop if a requested model is not installed instead of skipping it.",
    )
    args = parser.parse_args()

    if args.repeats < 2:
        parser.error("--repeats should be at least 2; 10 is recommended for the final thesis benchmark.")
    if args.cold_repeats < 1:
        parser.error("--cold-repeats must be at least 1.")
    if args.resource_repeats < 1:
        parser.error("--resource-repeats must be at least 1.")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive.")
    if not (0.0 <= args.temperature <= 2.0):
        parser.error("--temperature must be between 0 and 2.")
    if args.max_tokens < 20:
        parser.error("--max-tokens is too small for this benchmark; use at least 20.")
    return args


def choose_models(args: argparse.Namespace) -> List[str]:
    if args.only:
        return [args.only.strip()]
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    return list(MODELS_SERVER if args.profile == "server" else MODELS_LOCAL)


# -----------------------------------------------------------------------------
# Ollama endpoints
# -----------------------------------------------------------------------------
def normalize_openai_base(url: str) -> str:
    return url.rstrip("/") + "/"


def native_base_from_openai(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


def native_post(native_base: str, endpoint: str, payload: Dict[str, Any], timeout: float = 600.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        native_base + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama native API: {e}") from e


def show_model(native_base: str, model: str) -> Dict[str, Any]:
    try:
        data = native_post(native_base, "/api/show", {"model": model}, timeout=60.0)
        details = data.get("details", {}) or {}
        return {
            "model": model,
            "parameter_size": details.get("parameter_size", ""),
            "quantization_level": details.get("quantization_level", ""),
            "format": details.get("format", ""),
            "family": details.get("family", ""),
            "modified_at": data.get("modified_at", ""),
            "capabilities": data.get("capabilities", []),
        }
    except Exception as e:
        return {"model": model, "metadata_error": str(e)}


def unload_model(native_base: str, model: str) -> None:
    # Ollama documents keep_alive=0 as an immediate unload request.
    native_post(
        native_base,
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        timeout=120.0,
    )


def preload_model(native_base: str, model: str, seed: int) -> Dict[str, Any]:
    # Load the model before warm task timing. The warm-up response is discarded.
    return native_post(
        native_base,
        "/api/generate",
        {
            "model": model,
            "prompt": COLD_START_PROMPT,
            "stream": False,
            "keep_alive": WARM_KEEP_ALIVE,
            "options": {"num_predict": 2, "temperature": 0.0, "seed": seed},
        },
        timeout=600.0,
    )


# -----------------------------------------------------------------------------
# Objective response metrics
# -----------------------------------------------------------------------------
_PIPE_RUN_RE = re.compile(r"\|+")
_FORBIDDEN_FORMAT_RE = re.compile(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+|#+\s*\w", re.MULTILINE)

_QUESTION_START_RE = re.compile(
    r"^(?:"
    r"who|what|when|where|why|how|which|"
    r"can|could|would|will|do|does|did|is|are|was|were|have|has|had|"
    r"tell me|describe|explain|share|walk me through|give me"
    r")\b",
    re.IGNORECASE,
)

_FILLER_PREFIX_RE = re.compile(
    r"^(?:okay|alright|great|interesting|thanks(?: for sharing)?|sounds good|i see|that's great|that is great|wonderful)\s*[,!:-]?\s*",
    re.IGNORECASE,
)

_SECOND_QUESTION_CLAUSE_RE = re.compile(
    r"\b(?:and|or)\s+(?:who|what|when|where|why|how|which|can|could|would|do|does|did|is|are|was|were)\b",
    re.IGNORECASE,
)


def clean_text_for_words(text: str) -> str:
    return text.replace(SEPARATOR, " ").strip()


def count_question_units_heuristic(text: str) -> int:
    """Approximate number of question/request units.

    This is deliberately labelled a HEURISTIC. It catches both '?' questions
    and imperative interview requests such as 'Tell me about...' that do not
    contain a question mark. Human raters should resolve ambiguous cases.
    """
    cleaned = text.replace(SEPARATOR, "\n")
    segments = [s.strip() for s in re.split(r"[\n]+|(?<=[.!?])\s+", cleaned) if s.strip()]
    total = 0
    for segment in segments:
        qmarks = segment.count("?")
        stripped = _FILLER_PREFIX_RE.sub("", segment).strip()
        if qmarks > 0:
            units = qmarks
            # A single '?' may still contain two distinct interrogative clauses,
            # e.g. 'What happened and how did you solve it?'
            units += len(_SECOND_QUESTION_CLAUSE_RE.findall(stripped))
            total += units
        elif _QUESTION_START_RE.search(stripped):
            total += 1 + len(_SECOND_QUESTION_CLAUSE_RE.findall(stripped))
    return total


def response_metrics(text: str, expected_questions: int) -> Dict[str, Any]:
    raw = text or ""
    stripped = raw.strip()
    pipe_runs = _PIPE_RUN_RE.findall(stripped)
    malformed_pipe = any(run != SEPARATOR for run in pipe_runs)
    sep_count = stripped.count(SEPARATOR)
    sep_present = sep_count > 0
    sep_final = stripped.endswith(SEPARATOR)
    sep_exact_once_final = sep_count == 1 and sep_final and not malformed_pipe

    word_count = len(clean_text_for_words(stripped).split()) if stripped else 0
    question_marks = stripped.count("?")
    question_units = count_question_units_heuristic(stripped)
    expected_q_ok = question_units == expected_questions
    under_word_limit = word_count <= WORD_LIMIT
    forbidden_format = bool(_FORBIDDEN_FORMAT_RE.search(stripped))

    auto_instruction_pass = bool(
        stripped
        and sep_exact_once_final
        and under_word_limit
        and not forbidden_format
        and expected_q_ok
    )

    return {
        "word_count": word_count,
        "question_mark_count": question_marks,
        "question_units_heuristic": question_units,
        "expected_question_count": expected_questions,
        "question_count_pass": "YES" if expected_q_ok else "NO",
        "separator_count": sep_count,
        "separator_present": "YES" if sep_present else "NO",
        "separator_final": "YES" if sep_final else "NO",
        "malformed_pipe_run": "YES" if malformed_pipe else "NO",
        "separator_strict_pass": "YES" if sep_exact_once_final else "NO",
        "under_word_limit": "YES" if under_word_limit else "NO",
        "forbidden_formatting": "YES" if forbidden_format else "NO",
        "auto_instruction_pass": "YES" if auto_instruction_pass else "NO",
    }


# -----------------------------------------------------------------------------
# Model-specific prompt compatibility
# -----------------------------------------------------------------------------
def build_user_msg(model: str, raw_msg: str) -> tuple[str, str]:
    # Qwen3 supports hybrid thinking/non-thinking. The previous benchmark saw
    # empty visible content from thinking mode. Keep /no_think explicitly and
    # record that model-specific prompt adaptation in the output.
    if "qwen3" in model.lower():
        return raw_msg + " /no_think", "qwen3_no_think"
    return raw_msg, "standard"


# -----------------------------------------------------------------------------
# Timing calls
# -----------------------------------------------------------------------------
def run_streamed_task(
    client: OpenAI,
    model: str,
    task: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    user_msg, prompt_variant = build_user_msg(model, task["user_msg"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    first_token_ms: Optional[float] = None
    full_response = ""
    error = ""
    start = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
        )
        try:
            for chunk in stream:
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - start) * 1000.0
                    full_response += piece
        finally:
            try:
                stream.close()
            except Exception:
                pass
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    total_ms = (time.perf_counter() - start) * 1000.0
    metrics = response_metrics(full_response, task["expected_questions"])

    return {
        "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else -1,
        "total_ms": round(total_ms, 1),
        "prompt_variant": prompt_variant,
        "response": full_response.strip(),
        "error": error,
        **metrics,
    }


def measure_cold_load(native_base: str, model: str, repeat_idx: int, seed: int) -> Dict[str, Any]:
    unload_model(native_base, model)
    # Small pause gives the process a moment to release memory before reloading.
    time.sleep(0.5)

    wall_start = time.perf_counter()
    data = native_post(
        native_base,
        "/api/generate",
        {
            "model": model,
            "prompt": COLD_START_PROMPT,
            "stream": False,
            "keep_alive": 0,  # unload again after this cold measurement
            "options": {"num_predict": 2, "temperature": 0.0, "seed": seed},
        },
        timeout=1200.0,
    )
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    def ns_to_ms(value: Any) -> float:
        try:
            return float(value) / 1_000_000.0
        except Exception:
            return math.nan

    return {
        "model": model,
        "cold_repeat": repeat_idx,
        "error": "",
        "wall_total_ms": round(wall_ms, 1),
        "ollama_total_ms": round(ns_to_ms(data.get("total_duration")), 1),
        "ollama_load_ms": round(ns_to_ms(data.get("load_duration")), 1),
        "prompt_eval_ms": round(ns_to_ms(data.get("prompt_eval_duration")), 1),
        "generation_ms": round(ns_to_ms(data.get("eval_duration")), 1),
        "prompt_eval_tokens": data.get("prompt_eval_count", ""),
        "generated_tokens": data.get("eval_count", ""),
        "response": (data.get("response") or "").strip(),
    }


def measure_resource_probe(native_base: str, model: str, repeat_idx: int, seed: int) -> Dict[str, Any]:
    """Measure Ollama's native prompt/generation throughput on a fixed warm prompt."""
    data = native_post(
        native_base,
        "/api/generate",
        {
            "model": model,
            "prompt": RESOURCE_PROBE_PROMPT,
            "stream": False,
            "keep_alive": WARM_KEEP_ALIVE,
            "options": {"num_predict": 80, "temperature": 0.0, "seed": seed},
        },
        timeout=600.0,
    )

    def ns_to_s(value: Any) -> float:
        try:
            return float(value) / 1_000_000_000.0
        except Exception:
            return math.nan

    prompt_n = data.get("prompt_eval_count")
    eval_n = data.get("eval_count")
    prompt_s = ns_to_s(data.get("prompt_eval_duration"))
    eval_s = ns_to_s(data.get("eval_duration"))

    def rate(count: Any, seconds: float) -> float:
        try:
            return float(count) / seconds if seconds and seconds > 0 else math.nan
        except Exception:
            return math.nan

    return {
        "model": model,
        "resource_repeat": repeat_idx,
        "prompt_eval_tokens": prompt_n if prompt_n is not None else "",
        "generated_tokens": eval_n if eval_n is not None else "",
        "prompt_tokens_per_s": round(rate(prompt_n, prompt_s), 3) if not math.isnan(rate(prompt_n, prompt_s)) else "",
        "generation_tokens_per_s": round(rate(eval_n, eval_s), 3) if not math.isnan(rate(eval_n, eval_s)) else "",
        "load_ms": round(float(data.get("load_duration", 0)) / 1_000_000.0, 1) if data.get("load_duration") is not None else "",
        "total_ms": round(float(data.get("total_duration", 0)) / 1_000_000.0, 1) if data.get("total_duration") is not None else "",
        "response": (data.get("response") or "").strip(),
        "error": "",
    }


def gpu_ollama_memory_mb() -> Any:
    """Best-effort NVIDIA memory footprint for processes whose name contains 'ollama'."""
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        total = 0.0
        found = False
        for line in out.splitlines():
            if "ollama" not in line.lower():
                continue
            parts = [p.strip() for p in line.rsplit(",", 1)]
            if len(parts) == 2:
                try:
                    total += float(parts[1])
                    found = True
                except ValueError:
                    pass
        return round(total, 1) if found else ""
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Reproducibility metadata
# -----------------------------------------------------------------------------
def safe_cmd(command: List[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception:
        return ""


def environment_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "openai_compatible_base_url": args.base_url,
        "profile": args.profile,
        "repeats": args.repeats,
        "cold_repeats": args.cold_repeats,
        "resource_repeats": args.resource_repeats,
        "request_timeout_s": args.request_timeout,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "base_seed": args.seed,
        "separator": SEPARATOR,
        "word_limit": WORD_LIMIT,
        "system_prompt": SYSTEM_PROMPT,
        "tasks": TASKS,
    }
    meta["ollama_version"] = safe_cmd(["ollama", "--version"])
    if shutil.which("nvidia-smi"):
        meta["nvidia_smi"] = safe_cmd([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])
    else:
        meta["nvidia_smi"] = ""
    return meta


# -----------------------------------------------------------------------------
# CSV helpers / summaries
# -----------------------------------------------------------------------------
def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and float(v) >= 0]
    return statistics.mean(vals) if vals else math.nan


def sd_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and float(v) >= 0]
    return statistics.stdev(vals) if len(vals) >= 2 else math.nan


def median_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and float(v) >= 0]
    return statistics.median(vals) if vals else math.nan


def percentile_or_nan(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile, appropriate for transparent thesis reporting."""
    vals = sorted(float(v) for v in values if v is not None and float(v) >= 0)
    if not vals:
        return math.nan
    rank = max(1, math.ceil((p / 100.0) * len(vals)))
    return vals[min(rank - 1, len(vals) - 1)]


def pct_yes(rows: List[Dict[str, Any]], field: str) -> float:
    valid = [r for r in rows if not r.get("error")]
    if not valid:
        return math.nan
    return 100.0 * sum(1 for r in valid if r.get(field) == "YES") / len(valid)


def fmt_num(value: float, decimals: int = 1) -> Any:
    return "" if value is None or math.isnan(value) else round(value, decimals)


def build_summary(
    models: List[str],
    warm_rows: List[Dict[str, Any]],
    cold_rows: List[Dict[str, Any]],
    resource_rows: List[Dict[str, Any]],
    model_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summary = []
    for model in models:
        wr = [r for r in warm_rows if r["model"] == model]
        cr = [r for r in cold_rows if r["model"] == model]
        rr = [r for r in resource_rows if r["model"] == model and not r.get("error")]
        valid = [r for r in wr if not r.get("error") and r.get("response")]
        meta = model_meta.get(model, {})

        ft = [r["first_token_ms"] for r in valid]
        tot = [r["total_ms"] for r in valid]
        cold_load = [r["ollama_load_ms"] for r in cr]
        cold_total = [r["ollama_total_ms"] for r in cr]
        timeout_n = sum(1 for r in wr if "timeout" in str(r.get("error", "")).lower())
        failure_n = len(wr) - len(valid)
        prompt_tps = [r.get("prompt_tokens_per_s") for r in rr if r.get("prompt_tokens_per_s") not in ("", None)]
        generation_tps = [r.get("generation_tokens_per_s") for r in rr if r.get("generation_tokens_per_s") not in ("", None)]

        summary.append({
            "model": model,
            "parameter_size": meta.get("parameter_size", ""),
            "quantization_level": meta.get("quantization_level", ""),
            "warm_n": len(valid),
            "failed_n": failure_n,
            "failure_rate_pct": fmt_num(100.0 * failure_n / len(wr) if wr else math.nan),
            "timeout_n": timeout_n,
            "timeout_rate_pct": fmt_num(100.0 * timeout_n / len(wr) if wr else math.nan),
            "warm_first_token_mean_ms": fmt_num(mean_or_nan(ft)),
            "warm_first_token_sd_ms": fmt_num(sd_or_nan(ft)),
            "warm_first_token_median_ms": fmt_num(median_or_nan(ft)),
            "warm_first_token_p95_ms": fmt_num(percentile_or_nan(ft, 95)),
            "warm_total_mean_ms": fmt_num(mean_or_nan(tot)),
            "warm_total_sd_ms": fmt_num(sd_or_nan(tot)),
            "warm_total_median_ms": fmt_num(median_or_nan(tot)),
            "warm_total_p95_ms": fmt_num(percentile_or_nan(tot, 95)),
            "strict_separator_pass_pct": fmt_num(pct_yes(valid, "separator_strict_pass")),
            "question_count_pass_pct": fmt_num(pct_yes(valid, "question_count_pass")),
            "auto_instruction_pass_pct": fmt_num(pct_yes(valid, "auto_instruction_pass")),
            "mean_words": fmt_num(mean_or_nan([r["word_count"] for r in valid])),
            "cold_n": len(cr),
            "cold_load_mean_ms": fmt_num(mean_or_nan(cold_load)),
            "cold_load_sd_ms": fmt_num(sd_or_nan(cold_load)),
            "cold_total_mean_ms": fmt_num(mean_or_nan(cold_total)),
            "cold_total_sd_ms": fmt_num(sd_or_nan(cold_total)),
            "resource_n": len(rr),
            "prompt_tokens_per_s_mean": fmt_num(mean_or_nan(prompt_tps)),
            "generation_tokens_per_s_mean": fmt_num(mean_or_nan(generation_tps)),
            "gpu_ollama_memory_mb_snapshot": meta.get("gpu_ollama_memory_mb_snapshot", ""),
        })
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def response_id(model: str, task_id: str, repeat: int, response: str) -> str:
    payload = f"{model}|{task_id}|{repeat}|{response}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def make_blinded_rating_sheet(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    sheet = []
    for r in rows:
        if r.get("error") or not r.get("response"):
            continue
        sheet.append({
            "response_id": r["response_id"],
            "task_id": r["task_id"],
            "task_label": r["task_label"],
            "candidate_message": r["candidate_message"],
            "robot_response": r["response"],
            # These are intentionally blank. Two independent raters should fill
            # separate copies without knowing the model identity.
            "coherence_1_5": "",
            "naturalness_1_5": "",
            "task_appropriateness_1_5": "",
            "notes": "",
        })
    rng = random.Random(seed + 99991)
    rng.shuffle(sheet)
    return sheet


RATING_GUIDE = """HUMAN RATING GUIDE — use two independent raters if possible

Important:
- Do not show raters the model name.
- Each rater should fill a separate copy independently.
- Automatic properties (separator, length, question-count heuristic) are already measured by the script. Human ratings should focus on qualities that require judgment.

Coherence (1-5)
1 = incoherent/nonsensical
2 = major clarity or logical problems
3 = understandable but noticeably awkward or partially irrelevant
4 = clear and coherent with only minor issues
5 = fully clear, coherent, and contextually sensible

Naturalness (1-5)
1 = very robotic/inappropriate
2 = substantially unnatural
3 = acceptable but noticeably formulaic
4 = natural with minor awkwardness
5 = natural, professional interview language

Task appropriateness (1-5)
1 = fails the requested interview behavior
2 = mostly wrong behavior/topic
3 = partially appropriate
4 = appropriate with a minor issue
5 = fully appropriate for the candidate message and task

Reporting:
- If only one person rates the outputs, call it a "single-rater manual assessment", not a general human evaluation.
- With two raters, report agreement (e.g. weighted Cohen's kappa per ordinal scale) before averaging/combining ratings.
"""


# -----------------------------------------------------------------------------
# Main benchmark
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    requested_models = choose_models(args)
    client = OpenAI(base_url=normalize_openai_base(args.base_url), api_key="ollama", timeout=args.request_timeout)
    native_base = native_base_from_openai(args.base_url)

    print("Checking Ollama and installed models...")
    try:
        installed = [m.id for m in client.models.list().data]
    except Exception as e:
        raise SystemExit(f"Cannot reach Ollama at {args.base_url}: {e}")

    missing = [m for m in requested_models if m not in installed]
    if missing:
        print("\nMissing requested models:")
        for model in missing:
            print(f"  - {model}   (install: ollama pull {model})")
        if args.fail_on_missing:
            raise SystemExit("Stopping because --fail-on-missing was specified.")

    models = [m for m in requested_models if m in installed]
    if not models:
        raise SystemExit("None of the requested models is installed.")

    print("\nModels to benchmark:")
    for model in models:
        print(f"  - {model}")

    out_dir = Path(__file__).resolve().parent / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.only.replace(":", "-") if args.only else args.profile
    label_tag = f"_{args.label}" if args.label else ""
    stem = f"{model_tag}{label_tag}_{stamp}"

    warm_rows: List[Dict[str, Any]] = []
    cold_rows: List[Dict[str, Any]] = []
    resource_rows: List[Dict[str, Any]] = []
    model_meta: Dict[str, Dict[str, Any]] = {}

    for model_index, model in enumerate(models, start=1):
        print("\n" + "=" * 72)
        print(f"[{model_index}/{len(models)}] {model}")
        print("=" * 72)

        model_meta[model] = show_model(native_base, model)
        print(
            "Model metadata:",
            model_meta[model].get("parameter_size", "?"),
            model_meta[model].get("quantization_level", "?"),
        )

        # --- Separate cold-start phase ---
        print(f"Cold-load repetitions: {args.cold_repeats}")
        for cold_rep in range(1, args.cold_repeats + 1):
            try:
                row = measure_cold_load(
                    native_base,
                    model,
                    cold_rep,
                    seed=args.seed + model_index * 1000 + cold_rep,
                )
                cold_rows.append(row)
                print(
                    f"  cold {cold_rep}: load={row['ollama_load_ms']} ms, "
                    f"total={row['ollama_total_ms']} ms"
                )
            except Exception as e:
                cold_rows.append({
                    "model": model,
                    "cold_repeat": cold_rep,
                    "error": f"{type(e).__name__}: {e}",
                    "wall_total_ms": -1,
                    "ollama_total_ms": -1,
                    "ollama_load_ms": -1,
                    "prompt_eval_ms": -1,
                    "generation_ms": -1,
                    "prompt_eval_tokens": "",
                    "generated_tokens": "",
                    "response": "",
                })
                print(f"  cold {cold_rep}: ERROR {e}")

        # Explicitly preload before all quality/latency tasks.
        print("Preloading model for warm task measurements...")
        try:
            preload_model(native_base, model, args.seed + model_index * 1000 + 777)
            model_meta[model]["gpu_ollama_memory_mb_snapshot"] = gpu_ollama_memory_mb()
            model_meta[model]["ollama_ps_snapshot"] = safe_cmd(["ollama", "ps"])
        except Exception as e:
            print(f"WARNING: preload failed: {e}")

        # --- Repeated warm benchmark ---
        for repeat in range(1, args.repeats + 1):
            tasks = deepcopy(TASKS)
            # Use the SAME reproducible task order for every model in a given
            # repetition, so order effects are not confounded with model identity.
            rng = random.Random(args.seed + repeat)
            rng.shuffle(tasks)
            print(f"Warm repetition {repeat}/{args.repeats}")

            for order_in_repeat, task in enumerate(tasks, start=1):
                call_seed = args.seed + model_index * 1_000_000 + repeat * 100 + order_in_repeat
                result = run_streamed_task(
                    client,
                    model,
                    task,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=call_seed,
                )
                rid = response_id(model, task["id"], repeat, result["response"])
                row = {
                    "response_id": rid,
                    "model": model,
                    "parameter_size": model_meta[model].get("parameter_size", ""),
                    "quantization_level": model_meta[model].get("quantization_level", ""),
                    "repeat": repeat,
                    "order_in_repeat": order_in_repeat,
                    "seed": call_seed,
                    "temperature": args.temperature,
                    "task_id": task["id"],
                    "task_label": task["label"],
                    "expected_mode": task["expected_mode"],
                    "candidate_message": task["user_msg"],
                    **result,
                }
                warm_rows.append(row)

                status = "OK" if not result["error"] and result["response"] else "FAIL"
                print(
                    f"  [{status}] {task['id']:<18} "
                    f"TTFT={result['first_token_ms']:>8} ms  "
                    f"total={result['total_ms']:>8} ms  "
                    f"auto={result['auto_instruction_pass']}"
                )

        # --- Separate warm throughput/resource probe ---
        print(f"Resource probe repetitions: {args.resource_repeats}")
        for resource_rep in range(1, args.resource_repeats + 1):
            try:
                rr = measure_resource_probe(
                    native_base,
                    model,
                    resource_rep,
                    seed=args.seed + model_index * 10000 + resource_rep,
                )
                resource_rows.append(rr)
                print(
                    f"  resource {resource_rep}: generation={rr['generation_tokens_per_s']} tok/s, "
                    f"prompt={rr['prompt_tokens_per_s']} tok/s"
                )
            except Exception as e:
                resource_rows.append({
                    "model": model,
                    "resource_repeat": resource_rep,
                    "prompt_eval_tokens": "",
                    "generated_tokens": "",
                    "prompt_tokens_per_s": "",
                    "generation_tokens_per_s": "",
                    "load_ms": "",
                    "total_ms": "",
                    "response": "",
                    "error": f"{type(e).__name__}: {e}",
                })
                print(f"  resource {resource_rep}: ERROR {e}")

        # Release this model before moving to the next one.
        try:
            unload_model(native_base, model)
        except Exception:
            pass

    summary_rows = build_summary(models, warm_rows, cold_rows, resource_rows, model_meta)
    rating_sheet = make_blinded_rating_sheet(warm_rows, args.seed)

    raw_path = out_dir / f"benchmark_raw_{stem}.csv"
    cold_path = out_dir / f"benchmark_coldstart_{stem}.csv"
    summary_path = out_dir / f"benchmark_summary_{stem}.csv"
    resource_path = out_dir / f"benchmark_resource_{stem}.csv"
    rating_path = out_dir / f"benchmark_ratings_blinded_{stem}.csv"
    rating_guide_path = out_dir / f"benchmark_rating_guide_{stem}.txt"
    metadata_path = out_dir / f"benchmark_metadata_{stem}.json"

    write_csv(raw_path, warm_rows)
    write_csv(cold_path, cold_rows)
    write_csv(summary_path, summary_rows)
    write_csv(resource_path, resource_rows)
    write_csv(rating_path, rating_sheet)
    rating_guide_path.write_text(RATING_GUIDE, encoding="utf-8")

    metadata = environment_metadata(args)
    metadata["requested_models"] = requested_models
    metadata["benchmarked_models"] = models
    metadata["missing_models"] = missing
    metadata["model_metadata"] = model_meta
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    header = f"{'Model':<22} {'TTFT mean':>11} {'Total mean':>11} {'Auto pass':>10} {'Cold load':>11}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['model']:<22} "
            f"{str(row['warm_first_token_mean_ms']):>11} "
            f"{str(row['warm_total_mean_ms']):>11} "
            f"{str(row['auto_instruction_pass_pct']):>9}% "
            f"{str(row['cold_load_mean_ms']):>11}"
        )

    print("\nSaved:")
    for p in [raw_path, cold_path, summary_path, resource_path, rating_path, rating_guide_path, metadata_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
