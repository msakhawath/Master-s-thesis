"""
verify_frozen_hashes.py

Post-submission verification tool, created October 2026 for the examiner package.
It is NOT part of the original frozen experiment code.

What it does
------------
Every experiment in the thesis was "frozen" before it ran: the SHA-256 fingerprint
of each file it depended on was written to a manifest (see thesis Section 6.4 and
Appendix A.1). This script recomputes the fingerprint of every file listed in every
manifest and reports whether it still matches.

How to use
----------
1. Unzip frozen_sources_byte_exact.zip.
2. Run:   python verify_frozen_hashes.py <path-to-unzipped-frozen_sources>
   With no argument it looks for ./frozen_sources next to this script.
   You can also point it at a checkout of the GitHub repository.

Line endings
------------
The frozen files were hashed on Windows, where text files end lines with CRLF.
Git stores text files with LF, so a fresh checkout on macOS/Linux has LF line
endings and a different fingerprint even though the code is identical. For any
mismatch the script therefore also tests the file with LF converted back to CRLF
and reports "MATCH (after LF->CRLF)" when that explains the difference.
The byte-exact ZIP avoids the issue entirely.

Standard library only; Python 3.8 or newer.
"""
import hashlib
import json
import os
import re
import sys

MANIFESTS = [
    ("RQ2 (frozen 11 Aug 2026)", "rq3_results/rq3_frozen_rq2_manifest.json"),
    ("RQ3-A code (frozen 11 Aug 2026)", "rq3_results/rq3a_frozen_manifest.json"),
    ("RQ3-A result files", "rq3_results/rq3a_final_result_manifest.json"),
    ("RQ3-A statistics files", "rq3_results/rq3a_final_statistics_manifest.json"),
    ("RQ3-B code (frozen 11 Aug 2026)", "rq3_results/rq3b_frozen_manifest.json"),
    ("User study, pre-amendment (frozen 16 Aug 2026)",
     "archive_old/pre_amendment_20260819/study_frozen_manifest_PRE_AMENDMENT_20260819.json"),
    ("User study, post-amendment (re-frozen 20 Aug 2026)", "user_study/study_frozen_manifest.json"),
]
# In the pre-amendment manifest, main.py refers to the archived pre-amendment file.
OVERRIDES = {
    "archive_old/pre_amendment_20260819/study_frozen_manifest_PRE_AMENDMENT_20260819.json": {
        "main.py": "archive_old/pre_amendment_20260819/main_PRE_AMENDMENT_20260819.py",
    }
}
INSTRUMENT = "user_study/INSTRUMENT_FROZEN_HASHES.txt"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def entries(obj, key=None):
    """Yield (path, sha256) for every dict that carries a 'sha256' field."""
    if isinstance(obj, dict):
        if "sha256" in obj and key is not None:
            yield key, obj["sha256"].lower()
        for k, v in obj.items():
            yield from entries(v, k)


def resolve(root, path):
    p = path.replace("\\", "/")
    for cand in (p, "rq3_results/" + os.path.basename(p)):
        full = os.path.join(root, cand)
        if os.path.isfile(full):
            return full, cand
    return None, p


def check(full, expected):
    data = open(full, "rb").read()
    if sha256_bytes(data) == expected:
        return "MATCH"
    lf = data.replace(b"\r\n", b"\n")
    if sha256_bytes(lf.replace(b"\n", b"\r\n")) == expected:
        return "MATCH (after LF->CRLF)"
    return "DIFFERS"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "frozen_sources")
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}")
    total = ok = 0
    for label, mpath in MANIFESTS:
        mfull = os.path.join(root, mpath)
        print(f"\n=== {label}\n    manifest: {mpath}")
        if not os.path.isfile(mfull):
            print("    MANIFEST NOT FOUND here (it is included in frozen_sources_byte_exact.zip)")
            continue
        manifest = json.load(open(mfull, encoding="utf-8"))
        for fname, expected in entries(manifest):
            override = OVERRIDES.get(mpath, {}).get(fname.replace("\\", "/"))
            full, shown = resolve(root, override or fname)
            total += 1
            if full is None:
                status = "MISSING"
            else:
                status = check(full, expected)
            ok += status.startswith("MATCH")
            note = f"  <- {shown}" if override else ""
            print(f"    {status:24} {fname}{note}")
    # Separately frozen questionnaire instrument (plain-text hash file)
    print(f"\n=== Questionnaire instrument (frozen 16 Aug 2026)\n    hash file: {INSTRUMENT}")
    ifull = os.path.join(root, INSTRUMENT)
    if os.path.isfile(ifull):
        text = open(ifull, encoding="utf-8-sig").read()
        for path, digest in re.findall(r"^\s*(\S+\.\w+)\s*\n\s*SHA256:\s*([0-9A-Fa-f]{64})", text, re.M):
            rel = path.lstrip(".\\/").replace("\\", "/")
            full, shown = resolve(root, rel)
            total += 1
            status = "MISSING" if full is None else check(full, digest.lower())
            ok += status.startswith("MATCH")
            print(f"    {status:24} {rel}")
    print(f"\nSUMMARY: {ok} of {total} checks match.")
    print("Expected exception, explained in Some_notes.md: "
          "user_study/record_participant.py was changed during the study; "
          "the questionnaire file itself still matches.")


if __name__ == "__main__":
    main()
