"""
similarity.py — ASR-partial reuse compatibility scorer.

The score is used to decide whether speculative response work based on one ASR
hypothesis can be reused for a later hypothesis.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional
import os
import re
import numpy as np

try:
    import config as _cfg
except Exception:
    _cfg = None

TORCH_DEVICE = getattr(_cfg, "TORCH_DEVICE", "cpu")
USE_ONNX = bool(getattr(_cfg, "USE_ONNX", True))
ROOT_DIR = Path(
    getattr(_cfg, "ROOT_DIR", Path(__file__).resolve().parent)
)
ORT_PROVIDERS = getattr(
    _cfg, "ORT_PROVIDERS", ["CPUExecutionProvider"]
)
SIM_THREADS = int(
    getattr(_cfg, "SIM_THREADS", min(6, max(1, os.cpu_count() or 4)))
)
SIM_MAX_LENGTH = int(getattr(_cfg, "SIM_MAX_LENGTH", 128))
SIM_REQUIRE_BACKEND = bool(
    getattr(_cfg, "SIM_REQUIRE_BACKEND", True)
)

SIM_BACKEND = "onnx" if USE_ONNX else "torch"

_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_ONNX_DIR = ROOT_DIR / "onnx" / "all-MiniLM-L6-v2"
_ONNX_PATH = _ONNX_DIR / "model.onnx"

_backend: Optional[str] = None
_model = None
_tokenizer_onnx = None
_ort = None
_providers_selected = None
_last_error: Optional[str] = None

_last_utterance: Optional[str] = None
_MAX_CACHE = 512
_embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

_ws_re = re.compile(r"\s+")
_CONJ_END = {
    "and", "but", "or", "so", "because", "however", "therefore",
    "moreover", "also", "still", "yet", "well", "ok", "okay",
    "anyway", "now",
}
_WEAK_TAIL = ",;:…"


def _norm(s: str) -> str:
    return _ws_re.sub(" ", (s or "").strip())


def _strip_weak_tail(s: str) -> str:
    return s[:-1] if s and s[-1] in _WEAK_TAIL else s


def _cheap_similarity(a: str, b: str) -> Optional[float]:
    a = _strip_weak_tail(_norm(a))
    b = _strip_weak_tail(_norm(b))

    if not a or not b:
        return 0.0

    # Exact equality must be checked before incomplete-tail rules.
    if a == b:
        return 1.0

    at = a.lower().split()
    bt = b.lower().split()

    # Different hypotheses ending in a conjunction are treated conservatively.
    if at and at[-1] in _CONJ_END:
        return 0.0
    if bt and bt[-1] in _CONJ_END:
        return 0.0

    if min(len(a), len(b)) < 4:
        return 0.0

    if a.startswith(b) or b.startswith(a):
        shorter, longer = (
            (a, b) if len(a) <= len(b) else (b, a)
        )
        if len(shorter) / max(1, len(longer)) >= 0.8:
            return 0.95

    if at and bt:
        sa, sb = set(at), set(bt)
        union = sa | sb
        if union:
            j = len(sa & sb) / len(union)
            if j >= 0.95:
                return 0.95
            if j <= 0.10:
                return 0.0

    return None


def _np_l2_normalize(
    x: np.ndarray, axis: int = -1, eps: float = 1e-12
) -> np.ndarray:
    n = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _cos_sim_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(float(np.dot(a, b)), 0.0, 1.0))


def _ensure_backend() -> bool:
    global _backend, _model, _tokenizer_onnx
    global _ort, _providers_selected, _last_error

    if _backend is not None:
        return _model is not None

    if SIM_BACKEND == "onnx":
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            _ort = ort
            available = set(ort.get_available_providers())
            wanted = [
                p for p in ORT_PROVIDERS if p in available
            ] or ["CPUExecutionProvider"]

            so = ort.SessionOptions()
            so.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            so.intra_op_num_threads = max(1, SIM_THREADS)
            so.inter_op_num_threads = 1

            if not _ONNX_PATH.exists():
                raise FileNotFoundError(
                    f"ONNX similarity model not found: {_ONNX_PATH}"
                )

            _model = ort.InferenceSession(
                str(_ONNX_PATH),
                sess_options=so,
                providers=wanted,
            )
            _providers_selected = wanted

            tokenizer_source = (
                _ONNX_DIR
                if (_ONNX_DIR / "tokenizer.json").exists()
                else _MODEL_ID
            )
            _tokenizer_onnx = AutoTokenizer.from_pretrained(
                tokenizer_source
            )
            _backend = "onnx"
            _last_error = None
            return True

        except Exception as exc:
            _last_error = f"ONNX similarity backend failed: {exc!r}"
            _backend = None
            _model = None

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            _MODEL_ID, device=TORCH_DEVICE
        )
        if hasattr(_model, "max_seq_length"):
            _model.max_seq_length = SIM_MAX_LENGTH
        _backend = "torch"
        _last_error = None
        return True

    except Exception as exc:
        _backend = "none"
        _model = None
        _last_error = f"Torch similarity backend failed: {exc!r}"
        return False


def _require_backend() -> None:
    if _ensure_backend():
        return
    if SIM_REQUIRE_BACKEND:
        raise RuntimeError(
            f"No similarity backend is available. Last error: {_last_error}"
        )


def _embed_onnx(texts):
    enc = _tokenizer_onnx(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=SIM_MAX_LENGTH,
    )

    if not isinstance(enc.get("input_ids"), np.ndarray):
        enc = _tokenizer_onnx(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SIM_MAX_LENGTH,
        )
        input_ids = enc["input_ids"].cpu().numpy()
        attn_mask = enc["attention_mask"].cpu().numpy()
    else:
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

    input_names = {item.name for item in _model.get_inputs()}
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
    }

    if "token_type_ids" in input_names and "token_type_ids" in enc:
        token_type_ids = enc["token_type_ids"]
        if hasattr(token_type_ids, "cpu"):
            token_type_ids = token_type_ids.cpu().numpy()
        feeds["token_type_ids"] = token_type_ids

    output_names = [item.name for item in _model.get_outputs()]
    requested = (
        ["last_hidden_state"]
        if "last_hidden_state" in output_names
        else None
    )

    hidden = _model.run(requested, feeds)[0]
    mask = attn_mask[..., None]
    summed = (hidden * mask).sum(axis=1)
    denom = np.maximum(mask.sum(axis=1), 1.0)
    mean = summed / denom
    return np.asarray(_np_l2_normalize(mean, axis=1))


def _embed_torch(texts):
    embedding = _model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if hasattr(embedding, "detach"):
        return embedding.detach().cpu().numpy()
    return np.asarray(embedding)


def _encode_single(text: str):
    if _backend == "onnx":
        return _embed_onnx([text])[0]
    if _backend == "torch":
        return _embed_torch([text])[0]
    return None


def _get_embed(text: str):
    t = _norm(text)
    if not t:
        return None

    if t in _embed_cache:
        _embed_cache.move_to_end(t)
        return _embed_cache[t]

    _require_backend()
    if _model is None:
        return None

    emb = _encode_single(t)
    if emb is None:
        return None

    _embed_cache[t] = emb
    if len(_embed_cache) > _MAX_CACHE:
        _embed_cache.popitem(last=False)
    return emb


def compute_similarity(a: str, b: str) -> float:
    cheap = _cheap_similarity(a, b)
    if cheap is not None:
        return float(max(0.0, min(1.0, cheap)))

    ea = _get_embed(a)
    eb = _get_embed(b)
    if ea is None or eb is None:
        if SIM_REQUIRE_BACKEND:
            raise RuntimeError(
                f"Similarity embeddings unavailable. Last error: {_last_error}"
            )
        return 0.0

    return _cos_sim_np(ea, eb)


def compute_reuse_similarity(a: str, b: str) -> float:
    """Clearer thesis name for the same compatibility score."""
    return compute_similarity(a, b)


def is_repeat(utterance: str, threshold: float = 0.8) -> bool:
    global _last_utterance

    u = _norm(utterance)
    if not u:
        return False

    if _last_utterance is None:
        _last_utterance = u
        return False

    sim = compute_similarity(_last_utterance, u)
    _last_utterance = u
    return bool(sim >= threshold)


def reset_similarity_memory() -> None:
    global _last_utterance
    _last_utterance = None
    _embed_cache.clear()


def reset_memory() -> None:
    reset_similarity_memory()


def warm_up_similarity() -> bool:
    if not _ensure_backend():
        return False
    try:
        _ = _get_embed("warmup one")
        _ = _get_embed("warmup two")
        return True
    except Exception as exc:
        global _last_error
        _last_error = f"Similarity warm-up failed: {exc!r}"
        return False


def backend_info() -> dict:
    _ensure_backend()
    info = {
        "backend": _backend,
        "model_loaded": bool(_model is not None),
        "sim_threads": SIM_THREADS,
        "max_length": SIM_MAX_LENGTH,
        "require_backend": SIM_REQUIRE_BACKEND,
        "last_error": _last_error,
    }

    if _backend == "onnx":
        info["providers_selected"] = list(
            _providers_selected or []
        )
        info["onnx_path"] = str(_ONNX_PATH)
        info["onnx_path_exists"] = _ONNX_PATH.exists()
    elif _backend == "torch":
        info["torch_device"] = str(TORCH_DEVICE or "cpu")

    return info
