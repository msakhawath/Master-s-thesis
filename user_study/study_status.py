r"""Where does participant collection actually stand? Read-only.

Answers the question you cannot answer by looking at a folder: how many
participants COUNT, per condition, after exclusions - and is there a session log
sitting on disk that belongs to nobody.

    python user_study\study_status.py

Nothing is written. Safe to run at any time, including mid-session.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOC = ROOT / "user_study" / "allocation.csv"
MASTER = ROOT / "user_study" / "data" / "study_master.csv"
TARGET_PER_CONDITION = 10

# Session logs written before collection began are dev/pilot artefacts, not
# orphans. logs/ holds ~75 legacy CSVs inherited from the earlier system.
COLLECTION_STARTED = datetime(2026, 8, 16, tzinfo=timezone.utc)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_excluded(row: dict) -> bool:
    return (row.get("excluded") or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    alloc = read_csv(ALLOC)
    master = read_csv(MASTER)

    counted, excluded = [], []
    for r in master:
        (excluded if is_excluded(r) else counted).append(r)

    def n(rows, cond):
        return sum(1 for r in rows if (r.get("condition") or "").strip().upper() == cond)

    print("=" * 68)
    print("STUDY STATUS")
    print("=" * 68)

    print("\nCOUNTED TOWARD THE 20")
    for cond, label in (("A", "A  SLM multi-agent  "), ("B", "B  cloud-LLM legacy ")):
        have = n(counted, cond)
        bar = "#" * have + "." * max(0, TARGET_PER_CONDITION - have)
        flag = "  COMPLETE" if have >= TARGET_PER_CONDITION else ""
        print(f"  Condition {label} {have:2d}/{TARGET_PER_CONDITION}  [{bar}]{flag}")
        ids = sorted(r["participant_id"] for r in counted
                     if (r.get("condition") or "").strip().upper() == cond)
        if ids:
            print(f"      {', '.join(ids)}")

    if excluded:
        print("\nEXCLUDED - not counted, deliberately retained for the audit trail")
        for r in excluded:
            reason = (r.get("exclusion_reason") or "no reason recorded").strip()
            print(f"  {r['participant_id']:5s} cond {r.get('condition','?'):1s}  {reason}")
    else:
        print("\nEXCLUDED  none")

    # Questionnaires flagged incomplete are recorded but not yet usable.
    pending = [r for r in counted
               if (r.get("questionnaire_completed") or "").strip() not in {"1", "true", "yes"}]
    if pending:
        print("\nQUESTIONNAIRE INCOMPLETE - counted as run, not yet usable in analysis")
        for r in pending:
            note = (r.get("researcher_notes") or "").strip()
            print(f"  {r['participant_id']:5s} {note[:70]}")

    # Who is left, according to the (amended) allocation?
    recorded_ids = {r["participant_id"].upper() for r in master}
    remaining = [r for r in alloc if r["participant_id"].upper() not in recorded_ids]
    print(f"\nREMAINING PER ALLOCATION ({len(remaining)} slots)")
    for cond in ("A", "B"):
        ids = [r["participant_id"] for r in remaining
               if (r.get("condition") or "").strip().upper() == cond]
        print(f"  Condition {cond}: {len(ids):2d}  {', '.join(ids) if ids else '-'}")
    if remaining:
        nxt = remaining[0]
        entry = "main.py" if nxt["condition"].strip().upper() == "B" \
            else "rq3b_user_eval_main.py --condition vap_turngpt"
        print(f"\n  NEXT: {nxt['participant_id']}  Condition {nxt['condition']}  ->  {entry}")

    # Projection: does the plan still land on 10/10?
    for cond in ("A", "B"):
        proj = n(counted, cond) + sum(
            1 for r in remaining if (r.get("condition") or "").strip().upper() == cond)
        if proj != TARGET_PER_CONDITION:
            print(f"\n  WARNING: Condition {cond} projects to {proj}, not "
                  f"{TARGET_PER_CONDITION}. Allocation needs amending, or a "
                  f"replacement participant recruited for an excluded slot.")

    # Session logs on disk that belong to nobody.
    linked = {(r.get("raw_log_sha256") or "").strip().lower() for r in master}
    linked.discard("")
    orphans = []
    for pattern, sub in (("*_turntaking_*.csv", "logs"),
                         ("rq3b_vap_turngpt_*.csv", "rq3_results")):
        for p in (ROOT / sub).glob(pattern):
            mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
            if mtime < COLLECTION_STARTED:
                continue
            if sha256_file(p).lower() not in linked:
                orphans.append((mtime, p))

    # session_notes.csv explains logs that are deliberately not participant records:
    # pilots, aborted attempts, discarded runs. Without it every pilot shows up as an
    # anomaly forever and the check becomes noise, which is worse than no check.
    notes = {(r.get("sha256") or "").strip().lower(): r
             for r in read_csv(ROOT / "user_study" / "data" / "session_notes.csv")}

    explained, unexplained = [], []
    for mtime, p in orphans:
        entry = notes.get(sha256_file(p).lower())
        (explained if entry else unexplained).append((mtime, p, entry))

    print("\nSESSION LOGS NOT ATTACHED TO A PARTICIPANT RECORD")
    if unexplained:
        print("\n  UNEXPLAINED - a session ran and was never recorded. Resolve each one:")
        for mtime, p, _ in sorted(unexplained):
            print(f"    {mtime.astimezone():%Y-%m-%d %H:%M}  {p.relative_to(ROOT)}")
        print("\n    Either record it (counted), record it with --exclude (discarded),")
        print("    or add a row to user_study/data/session_notes.csv if it was a pilot.")
    else:
        print("  UNEXPLAINED: none - every session log is either recorded or documented")

    if explained:
        by_label: dict[str, int] = {}
        for _, _, entry in explained:
            by_label[entry["label"]] = by_label.get(entry["label"], 0) + 1
        print(f"\n  documented in session_notes.csv ({len(explained)}):")
        for label, count in sorted(by_label.items()):
            print(f"    {count:2d}  {label}")

    pending = [e for _, _, e in explained if "WHEN RECORDED" in (e.get("note") or "")]
    if pending:
        print("\n  ACTION PENDING - carry these into the participant record when run:")
        for e in pending:
            print(f"    {Path(e['path']).name}")

    print("\n" + "=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
