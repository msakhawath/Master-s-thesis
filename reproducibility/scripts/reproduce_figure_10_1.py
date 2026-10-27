"""
reproduce_figure_10_1.py

Post-submission reproduction script, created October 2026 from the preserved original
study data to reproduce Figure 10.1 and the candidate cut-off count reported in
Section 10.3.1 (4 of 138 Condition A turns). This file was not part of the original
frozen experiment code. The thesis (Appendix A.2) names make_figures.py as the
producer of Figure 10.1; that standalone script was not retained, and this file
replaces it.

Definitions (from user_study/ANALYSIS_PLAN.md, section 4, frozen 24 Aug 2026):
  detected speech per turn = acoustic_stop_ms - user_start_ms
  candidate gate-dropout turn = at least 3 words transcribed within a detected speech
  window that implies more than 5 words per second. A candidate shows that the
  microphone gate lost the voice; it does not by itself show the robot spoke over
  continuing speech (each candidate was validated manually by reading the next turn).

Sample: the ten included Condition A records in study_master.csv (not every file that
matches the Condition A file-name pattern; see thesis Appendix A.2).

Run from any folder, for example from the one that contains user_study/:
    python reproducibility/scripts/reproduce_figure_10_1.py [--show-text]
--show-text also prints each candidate turn and the turn after it, for the manual check.
Outputs go to analysis/results_reproduced/ (original results are not overwritten).
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _included_sessions import included, log_rows  # noqa: E402

OUT = os.path.join("analysis", "results_reproduced")


def main():
    show_text = "--show-text" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    sessions, candidates, total = [], [], 0
    for rec in included("A"):
        rows, _ = log_rows(rec)
        speech = []
        for i, row in enumerate(rows):
            try:
                dur = (float(row["acoustic_stop_ms"]) - float(row["user_start_ms"])) / 1000.0
            except (KeyError, ValueError):
                continue
            total += 1
            speech.append(dur)
            words = len((row.get("user_text") or "").split())
            if words >= 3 and dur > 0 and words / dur > 5:
                nxt = rows[i + 1].get("user_text", "") if i + 1 < len(rows) else ""
                candidates.append((rec["participant_id"], row["turn_index"], words, dur,
                                   row.get("user_text", ""), nxt))
        sessions.append((rec["participant_id"], st.median(speech), len(speech)))

    print(f"Included Condition A sessions: {len(sessions)}; turns: {total} (thesis: 138)")
    print(f"Candidate gate-dropout turns: {len(candidates)} = {100 * len(candidates) / total:.1f}% "
          f"(thesis: 4 of 138, 2.9%)")
    for pid, turn, words, dur, text, nxt in candidates:
        print(f"  {pid} turn {turn}: {words} words in {dur:.2f} s of detected speech")
        if show_text:
            print(f"      this turn : {text}\n      next turn : {nxt}")

    sessions.sort(key=lambda s: s[1])
    counts = {}
    for c in candidates:
        counts[c[0]] = counts.get(c[0], 0) + 1
    with open(os.path.join(OUT, "figure_10_1_values.csv"), "w", encoding="utf-8") as fh:
        fh.write("participant_id,median_detected_speech_s,turns,candidate_turns\n")
        for pid, med, n in sessions:
            fh.write(f"{pid},{med:.2f},{n},{counts.get(pid, 0)}\n")
    print("\nMedian detected speech per turn (s), sorted:")
    for pid, med, n in sessions:
        print(f"  {pid}  {med:5.1f}  ({n} turns, {counts.get(pid, 0)} candidates)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed: figure skipped (values saved to CSV).")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#c0392b" if counts.get(p) else "#7f8c8d" for p, _, _ in sessions]
    bars = ax.bar([p for p, _, _ in sessions], [m for _, m, _ in sessions], color=colours)
    for bar, (pid, _, _) in zip(bars, sessions):
        if counts.get(pid):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(counts[pid]), ha="center", color="#c0392b", fontsize=9)
    ax.set_ylabel("Median detected speech\nper turn (s)")
    ax.set_title("Answer style per Condition A session\n"
                 "(red = session contained candidate gate-dropout turns)", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "figure_10_1_reproduced.png")
    fig.savefig(path, dpi=150)
    print(f"\nFigure written to {path}")


if __name__ == "__main__":
    main()
