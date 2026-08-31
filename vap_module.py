"""
vap_module.py — thesis VAP runtime adapter

Changes:
- portable model paths from config.py
- Furhat Realtime speaker audio as robot channel by default
- real recent context (default 1.0 s), no artificial 2 s zero-padding
- explicit backend diagnostics and state-dict mismatch reporting
- warm_up() succeeds only if a real VAP backend executes
- heuristic fallback can be disabled for thesis experiments
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import config

try:
    from asr_module import get_user_audio
except Exception as exc:
    raise RuntimeError("Could not import asr_module.get_user_audio") from exc

_ROBOT_AUDIO_SOURCE = str(
    getattr(config, "VAP_ROBOT_AUDIO_SOURCE", "furhat_realtime")
).lower()

if _ROBOT_AUDIO_SOURCE == "furhat_realtime":
    try:
        from furhat_realtime_module import get_robot_audio
    except Exception as exc:
        raise RuntimeError(
            "VAP expects Furhat Realtime robot audio, but "
            "furhat_realtime_module.get_robot_audio could not be imported."
        ) from exc
elif _ROBOT_AUDIO_SOURCE == "legacy_tts":
    from tts_module import get_robot_audio
else:
    raise ValueError(
        "VAP_ROBOT_AUDIO_SOURCE must be 'furhat_realtime' or 'legacy_tts'"
    )

_SR = int(getattr(config, "SAMPLE_RATE", 16000))
_CONTEXT_SEC = float(getattr(config, "VAP_CONTEXT_SEC", 1.0))
_MIN_WIN_SEC = 0.30
_MAX_WIN_SEC = 4.00
_GATE_WIN_SEC = float(getattr(config, "VAP_GATE_WINDOW_SEC", 0.08))
_RMS_THRESH = float(getattr(config, "VAP_RMS_THRESHOLD", 0.010))

_TORCH_PT_PATH = Path(getattr(config, "VAP_TORCH_PT_PATH"))
_TORCH_CLASS = getattr(config, "VAP_TORCH_CLASS", "vap.model:VapGPT")
_DEVICE_PREF = str(getattr(config, "VAP_DEVICE", "auto")).lower()
_USE_HALF = bool(getattr(config, "VAP_TORCH_HALF", False))
_TORCH_NUM_THREADS = int(getattr(config, "VAP_TORCH_NUM_THREADS", 1))

_ONNX_PATH = Path(getattr(config, "VAP_ONNX"))
_ONNX_INTRA = int(getattr(config, "VAP_ONNX_INTRA", 4))
_ONNX_INTER = int(getattr(config, "VAP_ONNX_INTER", 1))
_ONNX_PROVIDERS = list(
    getattr(config, "VAP_ONNX_PROVIDERS", ["CPUExecutionProvider"])
)

_ALLOW_HEURISTIC = bool(
    getattr(config, "VAP_ALLOW_HEURISTIC_FALLBACK", False)
)
_FAIL_ON_STATE_MISMATCH = bool(
    getattr(config, "VAP_FAIL_ON_STATE_MISMATCH", False)
)

_torch = None
_ts = None
_model_torch = None
_ort = None
_ort_sess = None

_device = "cpu"
_dtype = None
_backend_name = "none"
_warmed = False
_ready = False
_forward_count = 0
_last_forward_ms = 0.0
_last_error: Optional[str] = None
_missing_keys: list[str] = []
_unexpected_keys: list[str] = []


def _to_float32_m1p1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    if x.dtype == np.int16:
        return np.clip(x.astype(np.float32) / 32768.0, -1.0, 1.0)
    return np.clip(x.astype(np.float32), -1.0, 1.0)


def _right_align(sig: np.ndarray, target_len: int) -> np.ndarray:
    sig = np.asarray(sig).reshape(-1)
    out = np.zeros(target_len, dtype=np.float32)
    if sig.size == 0:
        return out
    if sig.size >= target_len:
        return sig[-target_len:].astype(np.float32, copy=False)
    out[-sig.size:] = sig.astype(np.float32, copy=False)
    return out


def _rms(sig: np.ndarray) -> float:
    sig = np.asarray(sig)
    if sig.size == 0:
        return 0.0
    s = sig.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(s * s)))


def _stack_stereo(user, robot, target_len: int) -> np.ndarray:
    u = _right_align(_to_float32_m1p1(user), target_len)
    r = _right_align(_to_float32_m1p1(robot), target_len)
    return np.stack([u, r], axis=0)[None, ...].astype(np.float32)


def _choose_device(torch_module) -> str:
    if _DEVICE_PREF == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if _DEVICE_PREF == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("VAP_DEVICE='cuda' but CUDA is unavailable")
        return "cuda"
    if _DEVICE_PREF == "cpu":
        return "cpu"
    raise ValueError("VAP_DEVICE must be auto, cpu or cuda")


def _normalize_state_dict(state):
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    if not isinstance(state, dict):
        return state

    cleaned = {}
    for key, value in state.items():
        k = str(key)
        if "VAP.codebook" in k:
            continue
        if "vap_head.projection_head" in k:
            k = k.replace("vap_head.projection_head", "vap_head")
        if k.startswith("net."):
            k = k[4:]
        cleaned[k] = value
    return cleaned


def _real_backend_loaded() -> bool:
    return bool(_ts is not None or _model_torch is not None or _ort_sess is not None)


def _load_torch(force_reload: bool = False) -> bool:
    global _torch, _device, _dtype, _ts, _model_torch
    global _backend_name, _last_error, _missing_keys, _unexpected_keys

    if not force_reload and (_ts is not None or _model_torch is not None):
        return True

    try:
        import torch
        _torch = torch
    except Exception as exc:
        _last_error = f"torch import failed: {exc!r}"
        return False

    try:
        _torch.set_num_threads(max(1, _TORCH_NUM_THREADS))
    except Exception:
        pass

    try:
        _device = _choose_device(_torch)
    except Exception as exc:
        _last_error = str(exc)
        return False

    _dtype = (
        _torch.float16
        if (_device == "cuda" and _USE_HALF)
        else _torch.float32
    )

    if force_reload:
        _ts = None
        _model_torch = None

    if not _TORCH_PT_PATH.exists():
        _last_error = f"VAP .pt file not found: {_TORCH_PT_PATH}"
        return False

    # Try TorchScript first.
    try:
        candidate = _torch.jit.load(str(_TORCH_PT_PATH), map_location=_device)
        candidate.eval()
        _ts = candidate
        _backend_name = f"torchscript-{_device}"
        _last_error = None
        return True
    except Exception:
        _ts = None

    if not _TORCH_CLASS:
        _last_error = "VAP is not TorchScript and VAP_TORCH_CLASS is missing"
        return False

    # Then state_dict + supplied VapGPT class.
    try:
        mod_name, cls_name = str(_TORCH_CLASS).split(":", 1)
        module = importlib.import_module(mod_name)
        model_class = getattr(module, cls_name)
        model = model_class()

        state = _torch.load(
            str(_TORCH_PT_PATH),
            map_location=_device,
            weights_only=False,
        )
        state = _normalize_state_dict(state)

        incompatible = model.load_state_dict(state, strict=False)
        _missing_keys = list(getattr(incompatible, "missing_keys", []))
        _unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

        if _FAIL_ON_STATE_MISMATCH and (_missing_keys or _unexpected_keys):
            raise RuntimeError(
                f"state_dict mismatch: missing={_missing_keys[:20]}, "
                f"unexpected={_unexpected_keys[:20]}"
            )

        model.eval()
        model.to(device=_device, dtype=_dtype)
        _model_torch = model
        _backend_name = f"torch-{_device}"
        _last_error = None
        return True

    except Exception as exc:
        _model_torch = None
        _last_error = f"VAP Torch load failed: {exc!r}"
        return False


def _load_onnx() -> bool:
    global _ort, _ort_sess, _backend_name, _last_error

    if not _ONNX_PATH.exists():
        _last_error = f"VAP ONNX file not found: {_ONNX_PATH}"
        return False

    try:
        import onnxruntime as ort
        _ort = ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, _ONNX_INTRA)
        so.inter_op_num_threads = max(1, _ONNX_INTER)

        available = set(ort.get_available_providers())
        providers = [p for p in _ONNX_PROVIDERS if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        _ort_sess = ort.InferenceSession(
            str(_ONNX_PATH),
            sess_options=so,
            providers=providers,
        )
        _backend_name = "onnx"
        _last_error = None
        return True
    except Exception as exc:
        _ort = None
        _ort_sess = None
        _last_error = f"VAP ONNX load failed: {exc!r}"
        return False


def _load_backend() -> bool:
    global _torch
    try:
        import torch
        _torch = torch
    except Exception:
        _torch = None

    wants_cuda = (
        _DEVICE_PREF == "cuda"
        or (
            _DEVICE_PREF == "auto"
            and _torch is not None
            and _torch.cuda.is_available()
        )
    )
    if wants_cuda:
        return _load_torch(force_reload=True)

    # CPU: ONNX first, Torch second.
    if _load_onnx():
        return True
    return _load_torch(force_reload=True)


def _extract_torch_probs(p_now, p_future):
    if hasattr(p_now, "dim") and p_now.dim() == 3:
        p_now = p_now[:, -1, :]
    if hasattr(p_future, "dim") and p_future.dim() == 3:
        p_future = p_future[:, -1, :]

    vals = (
        float(p_now[0, 0].item()),
        float(p_future[0, 0].item()),
        float(p_now[0, 1].item()),
        float(p_future[0, 1].item()),
    )
    return tuple(max(0.0, min(1.0, v)) for v in vals)


def _forward_torch(x):
    global _last_forward_ms
    if _torch is None:
        raise RuntimeError("Torch backend is not initialized")

    t0 = time.perf_counter()
    xt = _torch.from_numpy(x).to(device=_device, dtype=_dtype)

    with _torch.inference_mode():
        if _ts is not None:
            out = _ts(xt)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                p_now, p_future = out[0], out[1]
            elif isinstance(out, dict):
                p_now = out.get("p_now", out.get("now"))
                p_future = out.get("p_future", out.get("future"))
                if p_now is None:
                    raise KeyError("TorchScript output has no p_now")
                if p_future is None:
                    p_future = p_now
            else:
                raise TypeError("Unsupported TorchScript VAP output")
        else:
            if _model_torch is None:
                raise RuntimeError("Torch model not loaded")
            out = _model_torch.probs(xt) if hasattr(_model_torch, "probs") else _model_torch(xt)
            p_now = out["p_now"]
            p_future = out["p_future"]

    result = _extract_torch_probs(p_now, p_future)
    _last_forward_ms = (time.perf_counter() - t0) * 1000.0
    return result


def _adapt_onnx_input(x):
    if _ort_sess is None:
        return x
    shape = _ort_sess.get_inputs()[0].shape
    if len(shape) < 3 or not isinstance(shape[-1], int) or shape[-1] <= 0:
        return x
    target = shape[-1]
    n = x.shape[-1]
    if n == target:
        return x
    if n > target:
        return x[..., -target:]
    return np.pad(x, ((0, 0), (0, 0), (target - n, 0)))


def _forward_onnx(x):
    global _last_forward_ms
    if _ort_sess is None:
        raise RuntimeError("ONNX backend is not initialized")

    t0 = time.perf_counter()
    wave = _adapt_onnx_input(x.astype(np.float32, copy=False))
    input_name = _ort_sess.get_inputs()[0].name
    outputs = _ort_sess.run(None, {input_name: wave})
    if len(outputs) < 2:
        raise RuntimeError("VAP ONNX must return p_now and p_future")

    p_now, p_future = outputs[0], outputs[1]
    if p_now.ndim == 3:
        p_now = p_now[:, -1, :]
    if p_future.ndim == 3:
        p_future = p_future[:, -1, :]

    vals = (
        float(p_now[0, 0]),
        float(p_future[0, 0]),
        float(p_now[0, 1]),
        float(p_future[0, 1]),
    )
    _last_forward_ms = (time.perf_counter() - t0) * 1000.0
    return tuple(max(0.0, min(1.0, v)) for v in vals)


def _forward_heuristic(x):
    global _last_forward_ms
    u = x[0, 0]
    r = x[0, 1]
    ru = _rms(u) + 1e-6
    rr = _rms(r) + 1e-6
    total = ru + rr
    pu_now = ru / total
    pr_now = rr / total
    pu_future = 0.2 + 0.6 * pu_now
    pr_future = 0.2 + 0.6 * pr_now
    _last_forward_ms = 0.1
    return pu_now, pu_future, pr_now, pr_future


def _forward(x):
    if _ts is not None or _model_torch is not None:
        return _forward_torch(x)
    if _ort_sess is not None:
        return _forward_onnx(x)
    if _ALLOW_HEURISTIC:
        return _forward_heuristic(x)
    raise RuntimeError(
        "No real VAP backend loaded and heuristic fallback is disabled. "
        f"Last error: {_last_error}"
    )


def warm_up(seconds: Optional[float] = None) -> bool:
    global _warmed, _ready, _forward_count, _last_error
    _warmed = True
    _ready = False
    _forward_count = 0

    if not _load_backend():
        return False

    try:
        n = int(round(max(_MIN_WIN_SEC, _CONTEXT_SEC) * _SR))
        dummy = np.zeros((1, 2, n), dtype=np.float32)
        _forward(dummy)
        _ready = True
        return True
    except Exception as exc:
        _last_error = f"warm-up forward failed: {exc!r}"
        _ready = False
        return False


def backend_info() -> Dict[str, object]:
    return {
        "backend": _backend_name,
        "real_backend_loaded": _real_backend_loaded(),
        "torch_available": bool(_torch is not None),
        "torchscript_loaded": bool(_ts is not None),
        "torch_module_loaded": bool(_model_torch is not None),
        "device": _device,
        "onnx_loaded": bool(_ort_sess is not None),
        "ready": _ready,
        "warmed": _warmed,
        "robot_audio_source": _ROBOT_AUDIO_SOURCE,
        "context_sec": _CONTEXT_SEC,
        "last_forward_ms": round(_last_forward_ms, 3),
        "missing_keys_count": len(_missing_keys),
        "unexpected_keys_count": len(_unexpected_keys),
        "missing_keys": _missing_keys[:20],
        "unexpected_keys": _unexpected_keys[:20],
        "last_error": _last_error,
        "torch_path": str(_TORCH_PT_PATH),
        "torch_path_exists": _TORCH_PT_PATH.exists(),
        "onnx_path": str(_ONNX_PATH),
        "onnx_path_exists": _ONNX_PATH.exists(),
        "heuristic_allowed": _ALLOW_HEURISTIC,
    }


def get_current(window_sec: Optional[float] = None) -> Dict[str, object]:
    global _forward_count, _ready, _last_error

    ctx = _CONTEXT_SEC if window_sec is None else float(window_sec)
    ctx = max(_MIN_WIN_SEC, min(_MAX_WIN_SEC, ctx))
    n = int(round(ctx * _SR))

    user = get_user_audio(ctx)
    robot = get_robot_audio(ctx)
    if isinstance(user, tuple):
        user = user[0]
    if isinstance(robot, tuple):
        robot = robot[0]

    x = _stack_stereo(user, robot, n)

    # Acoustic speaking gate uses only the most recent short slice, not the
    # whole VAP context.
    gate_n = max(1, int(round(min(ctx, _GATE_WIN_SEC) * _SR)))
    user_gate = _rms(x[0, 0, -gate_n:]) > _RMS_THRESH

    try:
        pn_u, pf_u, pn_r, pf_r = _forward(x)
        _forward_count += 1
        _ready = True
    except Exception as exc:
        _ready = False
        _last_error = f"inference failed: {exc!r}"
        raise

    return {
        "p_now_user": float(pn_u),
        "p_fut_user": float(pf_u),
        "p_now_robot": float(pn_r),
        "p_fut_robot": float(pf_r),
        "user_is_speaking": bool(user_gate),
        "ready": bool(_ready),
        "backend": _backend_name,
        "forward_ms": round(_last_forward_ms, 3),
    }
