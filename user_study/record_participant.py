#!/usr/bin/env python3
r"""Record participant feedback without modifying frozen study code.

Usage examples (run from the project root):

  python .\user_study\record_participant.py --participant P01 --condition A
  python .\user_study\record_participant.py --participant P02 --condition B

The script:
- detects the most likely raw session log (or accepts --log),
- collects the same 10-item Likert questionnaire for both conditions,
- collects open-ended feedback,
- collects researcher incident notes after the participant section,
- appends/updates one row in user_study/data/study_master.csv,
- writes a complete per-participant JSON record,
- hashes the raw log for traceability,
- never edits or renames the raw experimental log.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


QUESTIONNAIRE = [
    ("Q1", "The robot responded at an appropriate time after I finished speaking."),
    ("Q2", "I had enough time to finish my answers before the robot responded."),
    ("Q3", "The transitions between my turns and the robot's turns felt natural."),
    ("Q4", "I felt comfortable with the amount of time I had to wait for the robot's responses."),
    ("Q5", "The robot interrupted me unexpectedly."),
    ("Q6", "It was clear when it was my turn to speak and when it was the robot's turn."),
    ("Q7", "The conversation felt smooth and conversational."),
    ("Q8", "The interview felt natural and professional."),
    ("Q9", "The robot's questions and follow-up questions were relevant to my previous answers and to the interview."),
    ("Q10", "Overall, I was satisfied with the interview experience."),
    ("Q11", "The interview progressed through topics in a clear and sensible order."),
]

OPEN_QUESTIONS = [
    ("Q12", "What did you like most about the interaction?"),
    ("Q13", "What, if anything, felt unnatural, difficult, or frustrating?"),
    ("Q14", "Do you have any other comments about the interview system?"),
]

MASTER_FIELDS = [
    "participant_id",
    "condition",
    "recorded_at_local",
    "recorded_at_utc",
    "interview_completed",
    "questionnaire_completed",
    "raw_log",
    "raw_log_sha256",
    "raw_log_size_bytes",
    "raw_log_modified_at",
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q5_reversed", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11",
    "Q12_like_most",
    "Q13_unnatural_difficult_frustrating",
    "Q14_other_comments",
    "technical_incident",
    "researcher_intervention",
    "incident_description",
    "researcher_notes",
    # A discarded session is RECORDED, never deleted. Silently omitting a
    # session that happened is indistinguishable from dropping data whose
    # ratings were inconvenient; an examiner cannot tell the two apart, and
    # neither can you in six months. Rows with excluded=1 are retained in full
    # and skipped by analysis. Empty means counted.
    "excluded",
    "exclusion_reason",
]


@dataclass
class LogInfo:
    path: str
    sha256: str
    size_bytes: int
    modified_at: str


def project_root_from_script() -> Path:
    # .../project/user_study/record_participant.py -> .../project
    return Path(__file__).resolve().parent.parent


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def prompt_yes_no(message: str, default: Optional[bool] = None) -> bool:
    if default is True:
        suffix = " [Y/n]: "
    elif default is False:
        suffix = " [y/N]: "
    else:
        suffix = " [y/n]: "

    while True:
        raw = input(message + suffix).strip().lower()
        if not raw and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def prompt_likert(label: str, statement: str) -> int:
    print(f"\n{label}. {statement}")
    while True:
        raw = input("Your answer [1-5]: ").strip()
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if 1 <= value <= 5:
            return value
        print("Please enter a number from 1 to 5.")


def prompt_multiline(prompt: str) -> str:
    """Collect a free-text answer, guarding against the two failure modes seen live.

    Q12-Q14 are asked straight after eleven "[1-5]" Likert prompts, and five of the
    first six participants carried the habit over: P01 answered "4"/"4", P03 "3"/"4",
    P09 "1"/"2"/"1", while P05 and P11 left all three blank because the old prompt
    invited it. Only P07 wrote words. That left the qualitative strand - which the
    analysis plan codes into themes, and which supplies the thesis quotations - with
    one usable respondent out of six.

    Both guards are CONFIRMATIONS, never refusals. The consent form promises that a
    participant may skip any question, so a deliberate blank must always be possible.
    """
    print(f"\n{prompt}")
    print("Please answer in words - this one is not a 1-5 rating.")
    while True:
        answer = input("> ").strip()

        if not answer:
            if input("  Leave this question blank? [y/N]: ").strip().lower() == "y":
                return ""
            continue

        # A bare number here is almost certainly Likert carry-over, not an answer.
        if answer.replace(".", "", 1).isdigit():
            print("  That looks like a rating. This question asks for a sentence"
                  " in your own words.")
            if input("  Keep it as it is anyway? [y/N]: ").strip().lower() == "y":
                return answer
            continue

        return answer


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_local_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def log_info(path: Path, root: Path) -> LogInfo:
    stat = path.stat()
    try:
        display_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        display_path = str(path.resolve())
    return LogInfo(
        path=display_path,
        sha256=sha256_file(path),
        size_bytes=stat.st_size,
        modified_at=to_local_iso(stat.st_mtime),
    )


def newest(paths: Iterable[Path]) -> list[Path]:
    existing = [p for p in paths if p.is_file()]
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)


def detect_candidates(root: Path, participant: str, condition: str) -> list[Path]:
    if condition == "A":
        candidates: list[Path] = []
        # Primary expected location/pattern.
        candidates.extend((root / "rq3_results").glob(f"rq3b_vap_turngpt_{participant}_*.csv"))
        # Defensive fallback in case the participant-facing script writes elsewhere.
        candidates.extend(root.glob(f"rq3b_vap_turngpt_{participant}_*.csv"))
        # Last fallback: any participant-tagged vap_turngpt CSV below the project root.
        if not candidates:
            candidates.extend(root.rglob(f"rq3b_vap_turngpt_{participant}_*.csv"))
        return newest(dict.fromkeys(candidates))

    # Condition B filenames do not contain participant IDs in the frozen legacy controller.
    return newest((root / "logs").glob("*_turntaking_*.csv"))


def resolve_log(root: Path, participant: str, condition: str, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            raise FileNotFoundError(f"Raw log not found: {p}")
        return p.resolve()

    candidates = detect_candidates(root, participant, condition)
    if not candidates:
        raise FileNotFoundError(
            f"No raw log candidate found for Condition {condition}. "
            "Run the interview first or pass --log <path>."
        )

    print("\nDetected raw session log candidate:")
    first = candidates[0]
    print(f"  {first}")
    print(f"  modified: {to_local_iso(first.stat().st_mtime)}")

    if prompt_yes_no(f"Use this log for {participant}?", default=True):
        return first.resolve()

    print("\nRecent candidates:")
    for idx, p in enumerate(candidates[:10], start=1):
        print(f"  {idx:>2}. {p.name}   {to_local_iso(p.stat().st_mtime)}")
    print("   0. Enter a path manually")

    while True:
        raw = input("Choose log [0-10]: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            choice = -1
        if choice == 0:
            manual = Path(input("Path to raw CSV: ").strip().strip('"'))
            if not manual.is_absolute():
                manual = root / manual
            if manual.is_file():
                return manual.resolve()
            print("File not found.")
        elif 1 <= choice <= min(10, len(candidates)):
            return candidates[choice - 1].resolve()
        else:
            print("Please choose one of the listed numbers.")


def read_master(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_master(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(".backup.csv")
        shutil.copy2(path, backup)

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MASTER_FIELDS})
    tmp.replace(path)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def validate_participant(raw: str) -> str:
    participant = raw.strip().upper()
    if not re.fullmatch(r"P\d{2,3}", participant):
        raise argparse.ArgumentTypeError("Use a participant ID such as P01, P02, ... P20.")
    return participant


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one participant's post-interview study data.")
    parser.add_argument("--participant", required=True, type=validate_participant, help="Participant ID, e.g. P01")
    parser.add_argument("--condition", required=True, choices=["A", "B", "a", "b"], help="Study condition")
    parser.add_argument("--log", help="Explicit raw session-log path. If omitted, the script detects it.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing master/JSON record for this participant instead of refusing.",
    )
    parser.add_argument(
        "--exclude",
        metavar="REASON",
        help=("Record this session as EXCLUDED and skip the questionnaire. The session is "
              "kept in study_master.csv with excluded=1 so the audit trail is complete. "
              "State a reason that does NOT depend on the participant's ratings, e.g. "
              "'system crashed at turn 4', 'participant withdrew', 'closing detector ended "
              "the interview early'. Excluding because the scores looked wrong is biasing."),
    )
    args = parser.parse_args()

    participant = args.participant
    condition = args.condition.upper()
    root = project_root_from_script()
    data_dir = root / "user_study" / "data"
    master_path = data_dir / "study_master.csv"
    participant_dir = data_dir / participant
    json_path = participant_dir / "participant_record.json"

    existing_rows = read_master(master_path)
    existing_match = [r for r in existing_rows if r.get("participant_id", "").upper() == participant]
    if existing_match and not args.overwrite:
        print(f"ERROR: {participant} already exists in {master_path}.")
        print("Use --overwrite only if you intentionally need to correct that participant's record.")
        return 2

    print("=" * 72)
    print("RESEARCHER SETUP")
    print("=" * 72)
    print(f"Participant: {participant}")
    print(f"Condition:   {condition}")
    print(f"Project:     {root}")

    try:
        raw_log_path = resolve_log(root, participant, condition, args.log)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    info = log_info(raw_log_path, root)

    # Guard against attaching ONE session to TWO participants.
    #
    # Condition A logs carry the participant ID in the filename, so detection is
    # self-verifying. Condition B does not: the frozen legacy controller names its
    # log from a random uuid, so detection can only offer "the newest
    # logs/*_turntaking_*.csv" and the researcher confirms by eye. Recording out of
    # order, or after a session that crashed before writing, can therefore hand the
    # PREVIOUS participant's log to the current one. The existing check above only
    # catches the same participant twice, not the same log twice.
    clash = [r for r in existing_rows
             if (r.get("raw_log_sha256") or "").strip().lower() == info.sha256.lower()
             and r.get("participant_id", "").upper() != participant]
    if clash:
        owner = ", ".join(sorted(r.get("participant_id", "?") for r in clash))
        print(f"\nERROR: that raw log is already recorded against {owner}.")
        print(f"  {info.path}")
        print(f"  SHA-256: {info.sha256}")
        print("\nOne session cannot belong to two participants. Most likely the "
              "interview for this participant did not run, or did not write a log.")
        print("Re-run the interview, or pass --log <path> if you are certain which "
              "file is correct.")
        return 2

    # A log from long before now is usually the previous participant's.
    modified = datetime.fromtimestamp(raw_log_path.stat().st_mtime, timezone.utc)
    age_min = (datetime.now(timezone.utc) - modified).total_seconds() / 60.0
    if age_min > 120:
        print(f"\nWARNING: this log was last modified {age_min/60:.1f} hours ago.")
        if not prompt_yes_no("Is that really this participant's session?", default=False):
            print("Aborted. Re-run with --log <path>, or run the interview first.")
            return 2

    interview_completed = prompt_yes_no("Did the interview complete?", default=True)

    print("\nRaw log will remain untouched.")
    print(f"Mapped raw log: {info.path}")
    print(f"SHA-256:        {info.sha256}")

    if args.exclude:
        # Excluded session: keep the full provenance, skip the questionnaire.
        # The participant is not asked to rate an interview that will not count.
        print("\n" + "=" * 72)
        print("EXCLUDED SESSION")
        print("=" * 72)
        print(f"Reason: {args.exclude}")
        print("\nThis session will be recorded with excluded=1 and skipped by analysis.")
        print("The raw log is retained and left untouched. The questionnaire is skipped.")
        if not prompt_yes_no("Record this session as excluded?", default=False):
            print("Aborted. Nothing written.")
            return 1
        ratings = {code: "" for code, _ in QUESTIONNAIRE}
        q5_reversed = ""
        open_feedback = {code: "" for code, _ in OPEN_QUESTIONS}
        questionnaire_completed = False
        print("\nRESEARCHER INCIDENT RECORD")
        technical_incident = prompt_yes_no("Was there a technical incident?", default=True)
        researcher_intervention = prompt_yes_no("Did the researcher intervene?", default=False)
        incident_description = input("Brief incident description: ").strip()
        researcher_notes = input("Other researcher notes (optional): ").strip()
    else:
        input("\nHand the keyboard/device to the participant, then press Enter...")
        clear_screen()

        # Participant-facing section deliberately hides condition and raw-log details.
        print("=" * 72)
        print("POST-INTERVIEW QUESTIONNAIRE")
        print("=" * 72)
        print("Please rate each statement using:")
        print("  1 = Strongly disagree")
        print("  2 = Disagree")
        print("  3 = Neither agree nor disagree")
        print("  4 = Agree")
        print("  5 = Strongly agree")
        print("\nPlease answer based only on the interview you just completed.")

        ratings = {}
        for code, statement in QUESTIONNAIRE:
            ratings[code] = prompt_likert(code, statement)

        open_feedback = {}
        for code, question in OPEN_QUESTIONS:
            open_feedback[code] = prompt_multiline(f"{code}. {question}")

        questionnaire_completed = True
        print("\nThank you. Your questionnaire responses have been recorded.")
        input("Please return the keyboard/device to the researcher. Press Enter to continue...")
        clear_screen()

        print("=" * 72)
        print("RESEARCHER INCIDENT RECORD")
        print("=" * 72)
        technical_incident = prompt_yes_no("Was there a technical incident?", default=False)
        researcher_intervention = prompt_yes_no("Did the researcher intervene during the interview?", default=False)

        incident_description = ""
        if technical_incident or researcher_intervention:
            incident_description = input("Brief incident/intervention description: ").strip()
        researcher_notes = input("Other researcher notes (optional): ").strip()

    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    # Excluded sessions carry no ratings, so there is nothing to reverse-score.
    if not args.exclude:
        q5_reversed = 6 - ratings["Q5"]

    master_row: dict[str, object] = {
        "participant_id": participant,
        "condition": condition,
        "recorded_at_local": now_local.isoformat(timespec="seconds"),
        "recorded_at_utc": now_utc.isoformat(timespec="seconds"),
        "interview_completed": int(interview_completed),
        "questionnaire_completed": int(questionnaire_completed),
        "raw_log": info.path,
        "raw_log_sha256": info.sha256,
        "raw_log_size_bytes": info.size_bytes,
        "raw_log_modified_at": info.modified_at,
        **ratings,
        "Q5_reversed": q5_reversed,
        "Q12_like_most": open_feedback["Q12"],
        "Q13_unnatural_difficult_frustrating": open_feedback["Q13"],
        "Q14_other_comments": open_feedback["Q14"],
        "technical_incident": int(technical_incident),
        "researcher_intervention": int(researcher_intervention),
        "incident_description": incident_description,
        "researcher_notes": researcher_notes,
        "excluded": 1 if args.exclude else "",
        "exclusion_reason": args.exclude or "",
    }

    json_payload: dict[str, object] = {
        "participant_id": participant,
        "condition": condition,
        "recorded_at_local": master_row["recorded_at_local"],
        "recorded_at_utc": master_row["recorded_at_utc"],
        "interview": {
            "completed": interview_completed,
            "raw_log": asdict(info),
        },
        "questionnaire": {
            "scale": {
                "1": "Strongly disagree",
                "2": "Disagree",
                "3": "Neither agree nor disagree",
                "4": "Agree",
                "5": "Strongly agree",
            },
            "items": {
                code: {
                    "statement": statement,
                    "response": ratings[code],
                    **({"reverse_scored_value": q5_reversed} if code == "Q5" else {}),
                }
                for code, statement in QUESTIONNAIRE
            },
            "open_feedback": {
                code: {"question": question, "response": open_feedback[code]}
                for code, question in OPEN_QUESTIONS
            },
        },
        "researcher_record": {
            "technical_incident": technical_incident,
            "researcher_intervention": researcher_intervention,
            "incident_description": incident_description,
            "notes": researcher_notes,
        },
        "provenance": {
            "recording_script": str(Path(__file__).resolve().relative_to(root.resolve())),
            "raw_log_was_modified": False,
        },
    }

    # Update, rather than duplicate, when --overwrite was explicitly requested.
    if existing_match:
        updated_rows: list[dict[str, object]] = []
        replaced = False
        for row in existing_rows:
            if row.get("participant_id", "").upper() == participant and not replaced:
                updated_rows.append(master_row)
                replaced = True
            else:
                updated_rows.append(row)
        rows_to_write = updated_rows
    else:
        rows_to_write = [*existing_rows, master_row]

    save_json(json_path, json_payload)
    write_master(master_path, rows_to_write)

    print("\n" + "=" * 72)
    print("SAVED")
    print("=" * 72)
    print(f"Participant record: {json_path}")
    print(f"Master dataset:      {master_path}")
    print(f"Raw log:             {info.path}  (untouched)")
    print(f"Q5 reversed value:   {q5_reversed}  (stored separately; raw Q5 preserved)")
    print("\nNext participant: use a new participant ID and the assigned condition.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled. No participant record was intentionally finalized.")
        raise SystemExit(130)
