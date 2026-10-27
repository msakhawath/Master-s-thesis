"""Shared helper for the post-submission user-study reproduction scripts (created October 2026).

Loads the included sessions from user_study/data/study_master.csv and opens the
raw session log named in each record. The scripts find that folder by themselves (the project root, or study_data/ in the
private examiner package), so they can be run from any folder.
"""
import csv
import os

def _find_root():
    """Use the current folder if it holds user_study/, otherwise the folder two levels
    above this script (project root, or study_data/ in the private package)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.getcwd(), os.path.normpath(os.path.join(here, "..", ".."))):
        if os.path.isfile(os.path.join(cand, "user_study", "data", "study_master.csv")):
            return cand
    raise SystemExit("Cannot find user_study/data/study_master.csv; run from the folder that contains user_study/.")


ROOT = _find_root()
os.chdir(ROOT)   # all paths below are relative to the project root
MASTER = os.path.join("user_study", "data", "study_master.csv")


def included(condition):
    with open(MASTER, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    keep = [r for r in rows if r["condition"] == condition
            and str(r.get("excluded", "")).strip() in ("", "0", "False", "false")]
    return sorted(keep, key=lambda r: r["participant_id"])


def log_rows(record):
    path = record["raw_log"].replace("\\", "/")
    if not os.path.exists(path):
        for folder in ("rq3_results", "logs"):
            cand = os.path.join(folder, os.path.basename(path))
            if os.path.exists(cand):
                path = cand
                break
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh)), path
