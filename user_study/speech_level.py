r"""Is the microphone loud enough for the speaking gate to track a voice?

    python user_study\speech_level.py

The mirror of the room-noise check. Both arms decide "the user is speaking" with a bare RMS
threshold (VAD_RMS_THRESHOLD = 0.012 over 50 ms; VAP_RMS_THRESHOLD = 0.010 over 80 ms), with no
hysteresis and no speech model.

    Too much signal  -> the gate latches TRUE  -> the controller cannot yield -> deadlock
    Too little signal -> the gate drops to FALSE mid-answer -> acoustic_stop is set while the
                         participant is still talking, the hold expires, and the robot TAKES
                         THE FLOOR. The participant experiences this as being interrupted.

The second failure is invisible to preflight_check.py, which only measures the idle room.
Observed in P15 (2026-08-23): 4 of 13 turns registered under 300 ms of speech while
transcribing 3-6 words, implying up to 119 words/second - the gate saw a fraction of the
utterance. The same participants on 2026-08-16 showed none of this.

Read the RESULT: speech must sit comfortably ABOVE the threshold, not just cross it.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd
import config

SR = int(getattr(config, "SAMPLE_RATE", 16000))
DEV = getattr(config, "ASR_DEVICE_INDEX", None)
ASR_THR = float(getattr(config, "VAD_RMS_THRESHOLD", 0.012))
VAP_THR = float(getattr(config, "VAP_RMS_THRESHOLD", 0.010))
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
HOP = 0.05

PASSAGE = ("I completed my degree and then I worked on several projects, "
           "mostly involving data analysis and reporting for different teams.")


def main() -> int:
    blocks: list = []

    try:
        info = sd.query_devices(DEV, "input") if DEV is not None else sd.query_devices(kind="input")
        print(f"input device: {info.get('name','?')}")
    except Exception as exc:
        print(f"could not query device: {exc}")

    print(f"threshold: ASR {ASR_THR:.5f} | VAP {VAP_THR:.5f}\n")
    print("Sit exactly where the PARTICIPANT sits. Read this aloud at your normal")
    print("interview volume, repeatedly, for the whole countdown:\n")
    print(f'  "{PASSAGE}"\n')
    input("Press Enter, then start speaking...")

    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        blocksize=int(SR * HOP), device=DEV,
                        callback=lambda ind, f, t, s: blocks.append(
                            ind[:, 0].astype(np.float32) / 32768.0)):
        for i in range(int(SECS), 0, -1):
            print(f"  keep speaking... {i:2d}s ", end="\r")
            time.sleep(1.0)
    print(" " * 40)

    if not blocks:
        print("No audio captured.")
        return 1

    x = np.concatenate(blocks)
    w, h = int(SR * 0.05), int(SR * HOP)
    series = np.array([math.sqrt(float(np.mean(x[i:i + w] ** 2)))
                       for i in range(0, max(1, x.size - w), h)])

    # Speech has gaps. Judge the loud half - the actual voiced material - not the pauses.
    voiced = series[series >= np.percentile(series, 50)]
    above = float((series >= ASR_THR).mean() * 100)
    margin = float(np.median(voiced) / ASR_THR)

    # Longest continuous stretch reading as SILENT while you were speaking.
    best = run = 0
    for below in (series < ASR_THR):
        run = run + 1 if below else 0
        best = max(best, run)
    dropout_ms = best * HOP * 1000

    print("--- speech level ---")
    print(f"  median (voiced half) {np.median(voiced):.5f}")
    print(f"  p10 (quietest voiced) {np.percentile(voiced, 10):.5f}")
    print(f"  max                  {series.max():.5f}")
    print(f"  threshold            {ASR_THR:.5f}")
    print(f"\n  margin over threshold   {margin:.1f}x")
    print(f"  frames reading SPEAKING {above:.0f}%")
    print(f"  longest dropout         {dropout_ms:.0f} ms")

    print("\n--- verdict ---")
    ok = True
    if margin < 2.0:
        ok = False
        print("  FAIL: voice barely clears the threshold. The gate will lose it mid-answer")
        print("        and the robot will interrupt. RAISE THE MICROPHONE LEVEL.")
    elif margin < 3.0:
        print("  MARGINAL: only a small margin. Soft or trailing speech will be lost.")
    if dropout_ms >= 1000:
        ok = False
        print(f"  FAIL: {dropout_ms:.0f} ms continuous dropout while speaking - longer than the")
        print("        1000 ms hold, so the robot can take the floor mid-answer.")
    elif dropout_ms >= 500:
        print(f"  MARGINAL: {dropout_ms:.0f} ms dropout. Half the hold already consumed.")
    if above < 60:
        ok = False
        print(f"  FAIL: only {above:.0f}% of frames read as speaking.")

    if ok and margin >= 3.0 and dropout_ms < 500:
        print("  PASS: voice sits well clear of the threshold.")

    print("\n  How to fix a FAIL, in order of preference:")
    print("   1. Move the participant closer to the microphone.")
    print("   2. Raise the Windows input level for this device (Sound > Input > Properties).")
    print("   3. Re-check the room noise floor afterwards - raising gain raises BOTH.")
    print("      Run: python user_study\\preflight_check.py --condition A")
    print("      You need speech well above 0.012 AND the idle room below it.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
