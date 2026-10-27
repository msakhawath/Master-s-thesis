"""
reproduce_endpoint_residual.py

Post-submission reproduction script, created October 2026 from the preserved original
study data to reproduce the Condition A endpoint-delay figures in Sections 9.6, 10.3.1
and 10.3.2. This file was not part of the original frozen experiment code.

What it computes
  - endpoint decision delay per turn (endpoint_delay_ms), the median of the ten
    participant medians (thesis: 2719 ms) and the range of session medians
    (thesis: 2546-2967 ms);
  - the logged VAP forward-pass time (vap_forward_ms; thesis: about 56 ms);
  - the TurnGPT-related residual = mean endpoint delay - 1000 ms configured hold
    - mean VAP time.

Important: TurnGPT execution time was not logged as its own per-turn field. The
residual therefore contains TurnGPT processing plus any small controller-loop or
scheduling overhead that was not logged separately. See Some_notes.md.

Run from any folder, for example from the one that contains user_study/:
    python reproducibility/scripts/reproduce_endpoint_residual.py
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _included_sessions import included, log_rows  # noqa: E402

HOLD_MS = 1000.0   # RQ3B_USER_EVAL_MIN_GAP_MS in the frozen config.py


def main():
    delays, vap, medians = [], [], []
    for rec in included("A"):
        rows, _ = log_rows(rec)
        d = [float(r["endpoint_delay_ms"]) for r in rows if (r.get("endpoint_delay_ms") or "").strip()]
        v = [float(r["vap_forward_ms"]) for r in rows if (r.get("vap_forward_ms") or "").strip()]
        delays += d
        vap += v
        medians.append(st.median(d))
    mean_d, mean_v = st.mean(delays), st.mean(vap)
    residual = mean_d - HOLD_MS - mean_v
    print(f"Included Condition A sessions: {len(medians)}; turns with an endpoint delay: {len(delays)}")
    print(f"Median of participant medians: {st.median(medians):.0f} ms (thesis 9.6: 2719 ms)")
    print(f"Range of session medians:      {min(medians):.0f}-{max(medians):.0f} ms "
          f"(thesis 10.3.1: 2546-2967 ms)")
    print(f"Mean endpoint delay:           {mean_d:.1f} ms")
    print(f"Mean VAP forward time:         {mean_v:.1f} ms (thesis 10.3.2: about 56 ms)")
    print(f"Configured minimum hold:       {HOLD_MS:.0f} ms")
    print(f"TurnGPT-related residual:      {residual:.0f} ms (thesis 10.3.2: about 1631 ms)")
    print(f"Residual share of mean delay:  {100 * residual / mean_d:.1f}% "
          f"(thesis: about 62%; see Some_notes.md)")


if __name__ == "__main__":
    main()
