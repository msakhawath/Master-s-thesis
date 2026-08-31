"""
turngpt_module.py — thesis TurnGPT scorer

Key changes:
- validates trained <ts>/<s1>/<s2> tokens instead of adding untrained tokens
- builds recent multi-turn history in the same stream style used for training
- exposes raw neural P(<ts>) separately from controller-shaped score
- keeps get_turngpt_score() backward-compatible
"""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache
from typing import Dict, List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

try:
    import config
except Exception:
    config = None

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_DIR = os.path.join(HERE, "Trained_TurnGPT")
MODEL_DIR = str(getattr(config, "TURNGPT_DIR", DEFAULT_MODEL_DIR))

MAX_CTX_TOKENS = int(getattr(config, "TURNGPT_MAX_CTX_TOKENS", 256))
DEBUG = bool(getattr(config, "TURNGPT_DEBUG", False))
REQUIRE_MODEL = bool(getattr(config, "TURNGPT_REQUIRE_MODEL", True))
USE_VIRTUAL_FINAL_PUNCT = bool(
    getattr(config, "TURNGPT_VIRTUAL_FINAL_PUNCT", False)
)
USE_SHAPING = bool(getattr(config, "TURNGPT_USE_SHAPING", True))
_DEVICE_PREF = str(getattr(config, "TURNGPT_DEVICE", "auto")).lower()

TS_TOKEN = "<ts>"
S1 = "<s1>"
S2 = "<s2>"

_PARTIAL_CAP_YIELD = float(
    getattr(config, "TURNGPT_PARTIAL_CAP_YIELD", 0.40)
)
_PARTIAL_CAP_PREP = float(
    getattr(config, "TURNGPT_PARTIAL_CAP_PREPARE", 0.58)
)


def _select_device() -> str:
    if _DEVICE_PREF == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if _DEVICE_PREF == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("TURNGPT_DEVICE='cuda' but CUDA is unavailable")
        return "cuda"
    if _DEVICE_PREF == "cpu":
        return "cpu"
    raise ValueError("TURNGPT_DEVICE must be auto, cpu or cuda")


DEVICE = _select_device()
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_load_lock = threading.Lock()
_tok: Optional[GPT2TokenizerFast] = None
_model: Optional[GPT2LMHeadModel] = None
_ts_id: Optional[int] = None
_last_error: Optional[str] = None
_ws_re = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _ws_re.sub(" ", (text or "")).strip()


def _fail_or_floor(message: str) -> float:
    if REQUIRE_MODEL:
        raise RuntimeError(message)
    return 0.05


def _ensure_loaded() -> bool:
    global _tok, _model, _ts_id, _last_error

    if _tok is not None and _model is not None:
        return True

    with _load_lock:
        if _tok is not None and _model is not None:
            return True

        try:
            if not os.path.isdir(MODEL_DIR):
                raise FileNotFoundError(f"TurnGPT model dir not found: {MODEL_DIR}")

            tok = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
            tok.truncation_side = "left"
            model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

            if tok.eos_token is None:
                raise RuntimeError("TurnGPT tokenizer has no EOS token")

            tok.pad_token = tok.eos_token
            model.config.pad_token_id = tok.pad_token_id

            # These tokens must already have learned embeddings in the checkpoint.
            ids = {}
            for token in (TS_TOKEN, S1, S2):
                token_id = tok.convert_tokens_to_ids(token)
                if (
                    token_id is None
                    or token_id < 0
                    or token_id == tok.unk_token_id
                ):
                    raise RuntimeError(
                        f"Trained TurnGPT tokenizer is missing {token}"
                    )
                ids[token] = int(token_id)

            model.to(device=DEVICE, dtype=DTYPE).eval()

            dummy = tok(
                f"{S1} warm up ",
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_CTX_TOKENS,
            )
            with torch.inference_mode():
                _ = model(input_ids=dummy["input_ids"].to(DEVICE))

            _tok = tok
            _model = model
            _ts_id = ids[TS_TOKEN]
            _last_error = None
            return True

        except Exception as exc:
            _last_error = repr(exc)
            if DEBUG:
                print("[TurnGPT] load failed:", exc)
            return False


@lru_cache(maxsize=8192)
def _p_ts_next(prompt: str) -> float:
    if not _ensure_loaded():
        return _fail_or_floor(f"TurnGPT unavailable: {_last_error}")

    enc = _tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_CTX_TOKENS,
    )
    ids = enc["input_ids"].to(DEVICE)

    with torch.inference_mode():
        logits = _model(input_ids=ids).logits[0, -1]
        ts_logit = logits[_ts_id]
        prob = torch.exp(
            ts_logit - torch.logsumexp(logits, dim=-1)
        ).item()

    return float(max(0.0, min(1.0, prob)))


def _role_marker(role: str) -> Optional[str]:
    role = (role or "").lower().strip()
    if role == "user":
        return S1
    if role == "assistant":
        return S2
    return None


def _build_prompt(
    history: List[Dict[str, str]],
    latest_user: str,
    is_partial: bool,
) -> str:
    """
    Training style:
        <s1> utt <ts> <s2> utt <ts> ...

    At scoring time, every prior completed turn ends in <ts>, while the current
    user text deliberately does not. The model is therefore asked implicitly:
    is <ts> likely to be next?
    """
    latest_user = _norm(latest_user)
    parts: List[str] = []

    if is_partial:
        prior_messages = history
    else:
        current_user_index = None
        for i in range(len(history) - 1, -1, -1):
            if (history[i].get("role") or "").lower() == "user":
                current_user_index = i
                break
        prior_messages = (
            history[:current_user_index]
            if current_user_index is not None
            else history
        )

    for message in prior_messages:
        marker = _role_marker(message.get("role", ""))
        content = _norm(message.get("content", ""))
        if marker and content:
            parts.extend([marker, content, TS_TOKEN])

    if latest_user:
        parts.extend([S1, latest_user])

    return " ".join(parts).strip() + " "


def _prompt_token_count(prompt: str) -> Optional[int]:
    if not _ensure_loaded():
        return None
    token_ids = _tok(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_CTX_TOKENS,
    )["input_ids"]
    return len(token_ids)


_CONJ = {"and", "but", "so", "or", "because", "that", "which"}
_END_PUNCT = (".", "!", "?")
_WEAK_PUNCT = (",", ":", ";", "…")

_SHORT_LEN_CUTOFF = 3
_SHORT_LEN_SCALE = 0.25
_CONJ_END_SCALE = 0.35
_WEAK_PUNCT_SCALE = 0.50
STRONG_BOOST_KEEP = 0.65
DANGLING_SCALE = 0.50
SQUASH_POWER = 0.25


def _shape(raw_p: float, text: str, is_partial: bool = False) -> float:
    """
    Controller heuristic, not the neural TurnGPT probability.
    """
    s = _norm(text)
    words = s.split()
    n = len(words)
    last = words[-1].lower() if n else ""
    p = max(0.0, min(1.0, float(raw_p)))

    if n < _SHORT_LEN_CUTOFF:
        p *= _SHORT_LEN_SCALE
    if last in _CONJ:
        p *= _CONJ_END_SCALE
    if s.endswith(_WEAK_PUNCT):
        p *= _WEAK_PUNCT_SCALE
    if s.endswith(_END_PUNCT):
        p = 1.0 - (1.0 - p) * STRONG_BOOST_KEEP

    # The inherited r"\b\w$" condition treated nearly every normal word ending
    # as an unfinished token. Only an explicit trailing hyphen is kept.
    if is_partial and re.search(r"[A-Za-z]-$", s):
        p *= DANGLING_SCALE

    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    shaped = 1.0 / (
        1.0 + ((1.0 - p) / max(1e-6, p)) ** SQUASH_POWER
    )
    return float(max(0.0, min(1.0, shaped)))


def warm_up_turngpt() -> bool:
    if not _ensure_loaded():
        return False
    try:
        _ = _p_ts_next(f"{S1} warmup ")
        return True
    except Exception:
        return False


def turngpt_cache_clear() -> None:
    _p_ts_next.cache_clear()


def backend_info() -> Dict[str, object]:
    return {
        "loaded": bool(_tok is not None and _model is not None),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "model_dir": MODEL_DIR,
        "max_ctx_tokens": MAX_CTX_TOKENS,
        "require_model": REQUIRE_MODEL,
        "virtual_final_punct": USE_VIRTUAL_FINAL_PUNCT,
        "use_shaping": USE_SHAPING,
        "last_error": _last_error,
        "ts_token_id": _ts_id,
    }


def get_turngpt_details(
    history: List[Dict[str, str]],
    partial: Optional[str] = None,
    for_prepare: bool = False,
) -> Dict[str, object]:
    """
    Return raw neural P(<ts>) and controller-shaped score separately.
    """
    if not _ensure_loaded():
        raw = _fail_or_floor(f"TurnGPT unavailable: {_last_error}")
        return {
            "raw_probability": raw,
            "controller_input_probability": raw,
            "shaped_probability": raw,
            "score": raw,
            "is_partial": bool(partial),
            "for_prepare": bool(for_prepare),
            "cap": None,
            "prompt": "",
            "prompt_tokens": None,
            "text": "",
        }

    is_partial = partial is not None and bool(_norm(partial))

    if is_partial:
        latest = _norm(partial or "")
    else:
        latest = ""
        for msg in reversed(history):
            if (msg.get("role") or "").lower() == "user":
                latest = _norm(msg.get("content", ""))
                break
        if not latest:
            raw = _fail_or_floor("No user turn available for TurnGPT scoring")
            return {
                "raw_probability": raw,
                "controller_input_probability": raw,
                "shaped_probability": raw,
                "score": raw,
                "is_partial": False,
                "for_prepare": bool(for_prepare),
                "cap": None,
                "prompt": "",
                "prompt_tokens": None,
                "text": "",
            }

    latest_for_prompt = latest
    if (
        not is_partial
        and USE_VIRTUAL_FINAL_PUNCT
        and not latest.endswith(_END_PUNCT)
    ):
        latest_for_prompt = latest + "."

    prompt = _build_prompt(
        history,
        latest_for_prompt,
        is_partial=is_partial,
    )
    raw_probability = _p_ts_next(prompt)

    cap = _PARTIAL_CAP_PREP if for_prepare else _PARTIAL_CAP_YIELD
    controller_input = raw_probability
    if is_partial:
        controller_input = min(max(0.0, controller_input), cap)

    shaped = (
        _shape(controller_input, latest, is_partial=is_partial)
        if USE_SHAPING
        else controller_input
    )
    if is_partial:
        shaped = min(shaped, cap)

    shaped = float(max(0.0, min(1.0, shaped)))

    details = {
        "raw_probability": float(raw_probability),
        "controller_input_probability": float(controller_input),
        "shaped_probability": shaped,
        "score": shaped,
        "is_partial": bool(is_partial),
        "for_prepare": bool(for_prepare),
        "cap": float(cap) if is_partial else None,
        "prompt": prompt,
        "prompt_tokens": _prompt_token_count(prompt),
        "text": latest,
    }

    if DEBUG:
        print(
            "[TurnGPT] "
            f"partial={is_partial} prepare={for_prepare} "
            f"raw={raw_probability:.4f} shaped={shaped:.4f} "
            f"tokens={details['prompt_tokens']} text={latest[:80]!r}"
        )

    return details


def get_turngpt_score(
    history: List[Dict[str, str]],
    partial: Optional[str] = None,
    for_prepare: bool = False,
) -> float:
    """
    Backward-compatible controller API. For thesis logging use
    get_turngpt_details() so raw and shaped values remain distinguishable.
    """
    return float(
        get_turngpt_details(
            history,
            partial=partial,
            for_prepare=for_prepare,
        )["score"]
    )
