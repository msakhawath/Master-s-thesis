"""
Freeze both user-study arms before participant P01.

    python user_study\freeze_study_code.py --freeze     # write the manifest
    python user_study\freeze_study_code.py --verify     # check nothing changed

Mirrors the protocol used for RQ1/RQ2/RQ3: record the sha256 of every source
file either arm depends on, so it can be proven afterwards that all 20 sessions
ran on identical code. Run --verify after every few participants; if it fails
mid-study, the affected sessions are not comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "user_study" / "study_frozen_manifest.json"

SHARED = [
    "config.py",
    "asr_module.py",
]

CONDITION_A = [                      # SLM multi-agent arm
    "rq3b_user_eval_main.py",
    "rq3b_user_eval_policy.py",
    "multi_agent/agents.py",
    "multi_agent/graph.py",
    "multi_agent/state.py",
    "vap_module.py",
    "turngpt_module.py",
    "furhat_realtime_module.py",
]

CONDITION_B = [                      # legacy cloud-LLM arm
    "main.py",
    "llm_module.py",
    "tts_module.py",
    "furhat_module.py",
    "similarity.py",
    "logger.py",
]


def sha256(rel: str) -> tuple[str, int] | None:
    p = ROOT / rel
    if not p.exists():
        return None
    data = p.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def build() -> dict:
    import config

    entry: dict = {
        "status": "USER STUDY CODE FROZEN FOR PARTICIPANT COLLECTION",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "design": {
            "participants": 20,
            "per_condition": 10,
            "allocation": "between-subjects, block-randomised (user_study/allocation.csv)",
            "condition_A": "Qwen2.5:7B multi-agent, self-hosted on BHT Tesla P100",
            "condition_B": "GPT-4o-mini monolithic (legacy ChatGPT-based system)",
            "comparison_type": "system-level (arms differ in model, architecture, TTS, "
                               "robot interface and turn-taking)",
        },
        "runtime_settings": {
            "OPENAI_MODEL": getattr(config, "OPENAI_MODEL", None),
            "OPENAI_REQUEST_TIMEOUT": getattr(config, "OPENAI_REQUEST_TIMEOUT", None),
            "PREP_STALL_MS": getattr(config, "PREP_STALL_MS", None),
            "ENABLE_EARLY_LLM_PREFETCH": getattr(config, "ENABLE_EARLY_LLM_PREFETCH", None),
            "ENABLE_BARGE_IN": getattr(config, "ENABLE_BARGE_IN", None),
            "RQ3B_USER_EVAL_MIN_GAP_MS": getattr(config, "RQ3B_USER_EVAL_MIN_GAP_MS", 1000),
            "SIMPLE_SILENCE_MS": 2500,
        },
        "files": {},
        "missing": [],
    }

    for group, files in (("shared", SHARED), ("condition_A", CONDITION_A), ("condition_B", CONDITION_B)):
        entry["files"][group] = {}
        for rel in files:
            got = sha256(rel)
            if got is None:
                entry["missing"].append(rel)
                continue
            digest, size = got
            entry["files"][group][rel] = {"sha256": digest, "bytes": size}

    return entry


def do_freeze() -> int:
    entry = build()
    MANIFEST.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    n = sum(len(g) for g in entry["files"].values())
    print(f"Frozen {n} files -> {MANIFEST}")
    for group, files in entry["files"].items():
        print(f"\n  [{group}]")
        for rel in files:
            print(f"    {rel}")
    if entry["missing"]:
        print("\n  MISSING (not frozen):")
        for rel in entry["missing"]:
            print(f"    {rel}")
        return 1
    print("\nDo not edit any of these files until data collection is complete.")
    return 0


def do_verify() -> int:
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST} - run --freeze first.")
        return 1
    entry = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    print(f"Verifying against {MANIFEST.name} (frozen {entry.get('created_at')})\n")
    for group, files in entry["files"].items():
        print(f"  [{group}]")
        for rel, rec in files.items():
            got = sha256(rel)
            if got is None:
                print(f"    MISSING  {rel}")
                bad += 1
            elif got[0] != rec["sha256"]:
                print(f"    CHANGED  {rel}")
                bad += 1
            else:
                print(f"    ok       {rel}")

    import config
    print("\n  [runtime settings]")
    # Defaults must match build() exactly. They did not for
    # RQ3B_USER_EVAL_MIN_GAP_MS - build() recorded getattr(..., 1000) while verify
    # used getattr(..., None) - which reported a change that had not happened.
    _DEFAULTS = {"RQ3B_USER_EVAL_MIN_GAP_MS": 1000}
    for k, v in entry.get("runtime_settings", {}).items():
        if k == "SIMPLE_SILENCE_MS":
            now = v                       # constant in rq3b_user_eval_policy, not config
        else:
            now = getattr(config, k, _DEFAULTS.get(k))
        same = (now == v)
        print(f"    {'ok      ' if same else 'CHANGED '} {k}: frozen={v} now={now}")
        if not same:
            bad += 1

    print()
    if bad:
        print(f"{bad} DISCREPANCY(IES) - sessions run after the change are not comparable "
              "to those run before it. Record this in the incident log.")
        return 1
    print("VERIFY PASS - study code and settings unchanged since freeze.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    raise SystemExit(do_freeze() if a.freeze else do_verify())
