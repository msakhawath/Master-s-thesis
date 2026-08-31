r"""
Pre-session health check - run BEFORE each participant arrives.

    python user_study\preflight_check.py --condition A
    python user_study\preflight_check.py --condition B

Catches the failures that would otherwise be discovered with a participant in
the room: an expired or rate-limited API key, a dropped GPU port-forward, an
unreachable robot, a misconfigured microphone.

Exit code 0 = safe to run the session. Non-zero = do not start.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 - a failed check must never crash preflight
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, ok, detail))


# --------------------------------------------------------------- shared checks
def check_furhat():
    import furhat_realtime_module as FR
    FR.connect()
    try:
        FR.disconnect()
    except Exception:
        pass
    return True, f"{config.FURHAT_HOST}:{config.FURHAT_PORT}"


def check_assemblyai():
    key = getattr(config, "ASSEMBLYAI_API_KEY", "") or ""
    if not key or len(key) < 20:
        return False, "missing or implausible ASSEMBLYAI_API_KEY"
    return True, f"key present ({key[:4]}...{key[-4:]})"


def check_microphone():
    import sounddevice as sd
    idx = int(getattr(config, "ASR_DEVICE_INDEX", 1))
    dev = sd.query_devices(idx)
    if dev["max_input_channels"] < 1:
        return False, f"device {idx} '{dev['name']}' has no input channels"
    return True, f"[{idx}] {dev['name']}"


def check_noise_floor():
    """Reject a room whose noise floor trips the speaking gates.

    Both arms decide "the user is speaking" with a bare RMS threshold and no
    hysteresis - ASR.user_speaking() at VAD_RMS_THRESHOLD over 50 ms, and
    VAP's user_is_speaking at VAP_RMS_THRESHOLD over VAP_GATE_WINDOW_SEC.
    Neither uses a speech model. If ambient noise holds either gate True, the
    controller cannot yield and the turn stalls, so a noisy room silently
    destroys a session. check_microphone() only proves the device exists.

    The number that matters is not the mean level but the longest CONSECUTIVE
    run above threshold: a run longer than the hold can stall a turn on its own.
    Measured 2026-08-19: a degraded room gave 41% and a 2740 ms run; the same
    room once quietened gave 0.1% and 60 ms.
    """
    import math
    import time

    import numpy as np
    import sounddevice as sd

    sr = int(getattr(config, "SAMPLE_RATE", 16000))
    dev = getattr(config, "ASR_DEVICE_INDEX", None)
    hop = 0.02
    secs = float(getattr(config, "PREFLIGHT_NOISE_SECONDS", 10.0))

    gates = (
        ("ASR", 0.05, float(getattr(config, "VAD_RMS_THRESHOLD", 0.012))),
        ("VAP", float(getattr(config, "VAP_GATE_WINDOW_SEC", 0.08)),
         float(getattr(config, "VAP_RMS_THRESHOLD", 0.010))),
    )
    # The causal criterion is the RUN LENGTH, not the percentage. A gate trip
    # shorter than the hold is free: _settle_ok already requires gap_ms >= hold,
    # so the controller is waiting anyway. A trip only costs time if it is still
    # True at the moment gap_ms reaches the hold. The limit is therefore set at
    # half of LEGACY_MIN_HOLD_MS as a safety factor. The percentage is reported
    # for the record and fails only for a pathologically noisy room - the 23:52
    # pilot's room measured 41% with a 2740 ms run, a clean room 0.1% / 60 ms,
    # and a normal working room around 15% with sub-300 ms runs.
    hold_ms = float(getattr(config, "LEGACY_MIN_HOLD_MS", 1000))
    max_run_ms = float(getattr(config, "PREFLIGHT_NOISE_MAX_RUN_MS", hold_ms / 2.0))
    max_over_pct = float(getattr(config, "PREFLIGHT_NOISE_MAX_PCT", 35.0))

    blocks: list = []
    print(f"    measuring room noise for {secs:.0f}s - keep the room silent...")
    with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                        blocksize=int(sr * hop), device=dev,
                        callback=lambda ind, f, t, s: blocks.append(
                            ind[:, 0].astype(np.float32) / 32768.0)):
        time.sleep(secs)

    if not blocks:
        return False, "no audio captured - check the input device"
    x = np.concatenate(blocks)

    worst = []
    ok = True
    for name, win, thr in gates:
        w, h = int(sr * win), int(sr * hop)
        series = np.array([math.sqrt(float(np.mean(x[i:i + w] ** 2)))
                           for i in range(0, max(1, x.size - w), h)])
        over = series >= thr
        best = run = 0
        for flag in over:
            run = run + 1 if flag else 0
            best = max(best, run)
        pct, run_ms = float(over.mean() * 100.0), best * hop * 1000.0
        if pct > max_over_pct or run_ms > max_run_ms:
            ok = False
        worst.append(f"{name} {pct:.1f}%/{run_ms:.0f}ms")

    detail = ", ".join(worst)
    if not ok:
        return False, (f"{detail} - room too noisy "
                       f"(limit {max_run_ms:.0f}ms run / {max_over_pct:.0f}%)")
    return True, detail


def check_disk():
    out = ROOT / "rq3_results"
    out.mkdir(parents=True, exist_ok=True)
    probe = out / ".preflight_write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return True, str(out)


def check_config_invariants():
    stall_s = float(config.PREP_STALL_MS) / 1000.0
    api_s = float(config.OPENAI_REQUEST_TIMEOUT)
    if stall_s <= api_s:
        return False, f"PREP_STALL_MS ({stall_s}s) must exceed OPENAI_REQUEST_TIMEOUT ({api_s}s)"
    if getattr(config, "ENABLE_EARLY_LLM_PREFETCH", False):
        return False, "ENABLE_EARLY_LLM_PREFETCH must be False for the study"
    if getattr(config, "ENABLE_BARGE_IN", False):
        return False, "ENABLE_BARGE_IN must be False for the study"
    return True, f"stall {stall_s}s > api {api_s}s; prefetch off; barge-in off"


# ------------------------------------------------------- condition A (the SLM)
def check_ollama():
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="ollama", timeout=20.0)
    r = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        temperature=0.0,
        extra_body={"keep_alive": "30m"},
    )
    return bool(r.choices), "qwen2.5:7b reachable and warmed via 127.0.0.1:11435"


def check_vap_turngpt():
    import vap_module as VAP
    import turngpt_module as TG
    if not VAP.warm_up(seconds=float(getattr(config, "VAP_WARMUP_S", 4.0))):
        return False, f"VAP warm-up failed: {VAP.backend_info()}"
    if not TG.warm_up_turngpt():
        return False, f"TurnGPT warm-up failed: {TG.backend_info()}"
    return True, f"VAP {VAP.backend_info()}; TurnGPT {TG.backend_info()}"


# ------------------------------------------------- condition B (the cloud LLM)
def check_openai():
    from openai import OpenAI
    key = getattr(config, "OPENAI_API_KEY", "") or ""
    if not key.startswith(("sk-", "sk-proj-")):
        return False, "missing or malformed OPENAI_API_KEY"
    client = OpenAI(api_key=key, timeout=float(config.OPENAI_REQUEST_TIMEOUT), max_retries=0)
    r = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    return bool(r.choices), f"{config.OPENAI_MODEL} reachable, quota OK"


def check_openai_tts():
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=20.0, max_retries=0)
    with client.audio.speech.with_streaming_response.create(
        model=config.OPENAI_TTS_MODEL,
        voice=config.OPENAI_TTS_VOICE,
        input="test",
        response_format="pcm",
    ) as resp:
        chunk = next(resp.iter_bytes(chunk_size=1024), b"")
    return bool(chunk), f"{config.OPENAI_TTS_MODEL} / {config.OPENAI_TTS_VOICE}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["A", "B"],
                    help="A = SLM multi-agent system; B = cloud-LLM legacy system")
    args = ap.parse_args()

    print(f"\nPREFLIGHT - condition {args.condition}\n" + "=" * 60)

    check("config invariants", check_config_invariants)
    check("disk writable", check_disk)
    check("microphone", check_microphone)
    check("room noise floor", check_noise_floor)
    check("AssemblyAI key", check_assemblyai)
    check("Furhat", check_furhat)

    if args.condition == "A":
        check("Ollama / qwen2.5:7b (P100 port-forward)", check_ollama)
        check("VAP + TurnGPT", check_vap_turngpt)
    else:
        check("OpenAI chat (gpt-4o-mini)", check_openai)
        check("OpenAI TTS", check_openai_tts)

    print()
    failed = 0
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:42s} {detail}")
        if not ok:
            failed += 1

    print("=" * 60)
    if failed:
        print(f"{failed} CHECK(S) FAILED - do NOT start the session.\n")
        return 1
    print("ALL CHECKS PASSED - safe to run the session.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
