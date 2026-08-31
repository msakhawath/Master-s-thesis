r"""Retroactively exclude a participant who has ALREADY been recorded.

    python user_study\exclude_recorded_session.py --participant P07 \
        --reason "microphone dropped out from turn 6; robot took the floor mid-answer"

Use this only when the session was already written to study_master.csv. For a session that has
not been recorded yet, use record_participant.py --exclude instead.

WHY THIS IS A SEPARATE TOOL, AND A CAREFUL ONE
    record_participant.py --overwrite would replace the row and destroy the responses. That is
    the wrong operation: an excluded participant's data must be RETAINED and marked, never
    deleted, or the exclusion is indistinguishable from the data never having existed.

    More importantly, a retroactive exclusion is made AFTER the ratings are visible. That is
    the most bias-prone decision in the study: excluding a participant whose scores you dislike,
    for a reason you rationalise afterwards, is undetectable in the final dataset. So this tool
    records that the decision was retroactive, and when it was made, so the timing is auditable.

    The reason must be a TECHNICAL FAULT, not the behaviour under measurement. "Microphone
    failed", "backend errored", "session crashed" are faults. "The robot interrupted", "it was
    slow", "the answers were poor", "the ratings look wrong" are your dependent variables -
    Q1-Q7 measure timing and interruption, Q9 measures adaptivity. Discarding sessions on those
    grounds trims the low end of your own outcome measure.

Nothing is deleted. The row keeps every response; analysis skips rows with excluded=1.
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "user_study" / "data" / "study_master.csv"

TRUTHY = {"1", "true", "yes", "y"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, help="Participant ID already in study_master.csv")
    ap.add_argument("--reason", required=True,
                    help="TECHNICAL fault that invalidates the session. Not the ratings, "
                         "not the robot's conversational behaviour.")
    ap.add_argument("--undo", action="store_true",
                    help="Reverse a previous exclusion (restores the participant to counted).")
    args = ap.parse_args()

    pid = args.participant.strip().upper()

    if not MASTER.is_file():
        print(f"ERROR: {MASTER} not found.")
        return 2

    with MASTER.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("ERROR: study_master.csv is empty.")
        return 2

    fields = list(rows[0].keys())
    for needed in ("excluded", "exclusion_reason"):
        if needed not in fields:
            fields.append(needed)
            for r in rows:
                r.setdefault(needed, "")

    target = [r for r in rows if r["participant_id"].strip().upper() == pid]
    if not target:
        print(f"ERROR: {pid} is not in study_master.csv.")
        print("  Recorded: " + ", ".join(sorted(r['participant_id'] for r in rows)))
        print("  For a session that was never recorded, use "
              "record_participant.py --exclude instead.")
        return 2
    row = target[0]
    already = (row.get("excluded") or "").strip().lower() in TRUTHY

    if args.undo:
        if not already:
            print(f"{pid} is not currently excluded. Nothing to undo.")
            return 0
    elif already:
        print(f"{pid} is already excluded: {row.get('exclusion_reason','')}")
        return 0

    stamp = datetime.now().astimezone().isoformat(timespec="seconds")

    print("=" * 72)
    print(f"{'UNDO EXCLUSION' if args.undo else 'RETROACTIVE EXCLUSION'}: {pid}")
    print("=" * 72)
    print(f"  condition:        {row.get('condition','?')}")
    print(f"  recorded at:      {row.get('recorded_at_local','?')}")
    print(f"  raw log:          {row.get('raw_log','?')}")
    ratings = " ".join(f"{q}={row.get(q) or '-'}" for q in
                       ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11"))
    print(f"  responses:        {ratings}")
    print(f"\n  reason:           {args.reason}")

    if not args.undo:
        print("\n  This decision is being made AFTER the responses above were visible.")
        print("  That is recorded. The reason must be a technical fault, not the ratings")
        print("  and not the robot's conversational behaviour (Q1-Q7 measure timing and")
        print("  interruption; Q9 measures adaptivity - those are your data, not defects).")
        print("\n  Excluding CONSUMES this participant ID. To keep 10/10, recruit a")
        print("  replacement with the next free ID of the same parity (odd->A, even->B).")

    if input("\nProceed? [y/N]: ").strip().lower() != "y":
        print("Aborted. Nothing changed.")
        return 1

    backup = MASTER.with_name(f"study_master.pre_exclude_{pid}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    shutil.copy2(MASTER, backup)

    note_prefix = "UN-EXCLUDED" if args.undo else "RETROACTIVELY EXCLUDED"
    note = (f"{stamp[:10]}: {note_prefix} - {args.reason} "
            f"(decided after the responses were visible; all responses retained)")
    cur = (row.get("researcher_notes") or "").strip()
    row["researcher_notes"] = (cur + " " if cur else "") + note
    row["excluded"] = "" if args.undo else "1"
    row["exclusion_reason"] = "" if args.undo else f"[retroactive {stamp[:10]}] {args.reason}"

    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    MASTER.write_text(out.getvalue(), encoding="utf-8-sig", newline="")

    counted = [r for r in rows if (r.get("excluded") or "").strip().lower() not in TRUTHY]
    print(f"\n  backup:  {backup.name}")
    print(f"  {pid} is now {'COUNTED' if args.undo else 'EXCLUDED'}. Responses retained in full.")
    for cond in ("A", "B"):
        n = sum(1 for r in counted if (r.get("condition") or "").strip().upper() == cond)
        print(f"  Condition {cond}: {n}/10 counted")
    print("\n  Run: python user_study\\study_status.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
