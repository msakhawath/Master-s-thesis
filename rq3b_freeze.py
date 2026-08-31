from pathlib import Path
from datetime import datetime
import hashlib
import json
import config

FILES = [
    Path("rq3b_main.py"),
    Path("rq3b_policy.py"),
    Path("multi_agent/agents.py"),
    Path("multi_agent/graph.py"),
    Path("multi_agent/state.py"),
    Path("asr_module.py"),
    Path("vap_module.py"),
    Path("turngpt_module.py"),
    Path("furhat_realtime_module.py"),
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

missing = [str(p) for p in FILES if not p.exists()]
if missing:
    raise RuntimeError("Missing files:\n" + "\n".join(missing))

manifest = {
    "status": "RQ3-B CODE FROZEN FOR FINAL COLLECTION",
    "created_at": datetime.now().astimezone().isoformat(),
    "design": {
        "matched_repetitions": 20,
        "conditions": ["simple", "vap_turngpt"],
        "trials_per_condition": 20,
        "total_trials": 40,
        "inferential_unit": "matched scripted phrase repetition",
        "simple_silence_ms": 2500,
        "partial_asr_llm_prefetch": False,
        "same_multi_agent_backend": True,
        "same_model": "qwen2.5:7b",
    },
    "predictive_policy": {
        "min_gap_ms": int(config.MIN_GAP_TIME),
        "vap_pnow_yield_threshold": float(config.VAP_PNOW_YIELD_THRESHOLD),
        "vap_pfut_yield_threshold": float(config.VAP_PFUT_YIELD_THRESHOLD),
        "user_low_now_for_yield": float(config.USER_LOW_NOW_FOR_YIELD),
        "user_low_fut_for_yield": float(config.USER_LOW_FUT_FOR_YIELD),
        "turngpt_yield_timeouts": [
            [float(p), int(ms)] for p, ms in config.TURNGPT_YIELD_TIMEOUTS
        ],
    },
    "files": {
        str(p): {
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in FILES
    },
}

out = Path("rq3_results/rq3b_frozen_manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("RQ3-B FROZEN FOR FINAL COLLECTION")
print(out)
for p, info in manifest["files"].items():
    print(f"{p}: {info['sha256']}")
