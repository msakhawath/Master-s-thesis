r"""Formative user-study analysis. READ-ONLY on study data.

    python analysis\run_user_study_analysis.py

Implements user_study/ANALYSIS_PLAN.md exactly. Writes only into analysis/results/.
Touches nothing under user_study/data/, rq3_results/ or logs/.

Key commitments from the frozen plan:
  * unit of analysis is the PARTICIPANT, never the turn (no pseudo-replication)
  * P09 questionnaire is MISSING (A n=9 for items); P09 stays in objective analysis (n=10)
  * Q5 reverse-scored; medians/IQR primary; Mann-Whitney exploratory; rank-biserial emphasised
  * Holm across the 11 items, reported next to raw p
  * no composite questionnaire score
  * qualitative is available-case; no between-condition prevalence claims
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "user_study" / "data" / "study_master.csv"
OUT = ROOT / "analysis" / "results"
OUT.mkdir(parents=True, exist_ok=True)

LIKERT = [f"Q{i}" for i in range(1, 12)]
OPEN = ["Q12_like_most", "Q13_unnatural_difficult_frustrating", "Q14_other_comments"]
TRUTHY = {"1", "true", "yes", "y"}

QUESTION_TEXT = {
    "Q1": "The robot responded at an appropriate time after I finished speaking",
    "Q2": "The robot allowed me enough time to finish my answers",
    "Q3": "The transitions between my turns and the robot's turns felt natural",
    "Q4": "The waiting time before the robot responded felt comfortable",
    "Q5": "I was interrupted by the robot unexpectedly (REVERSE-SCORED)",
    "Q6": "It was clear when it was my turn to speak",
    "Q7": "The interview felt smooth and conversational",
    "Q8": "The interviewer felt natural and professional",
    "Q9": "The follow-up questions were appropriate to what I said",
    "Q10": "Overall, I was satisfied with this interaction",
    "Q11": "Topic order / coverage",
}


def read_csv(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(v):
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def iqr(xs):
    if len(xs) < 2:
        return None, None
    q1, q3 = np.percentile(xs, [25, 75])
    return float(q1), float(q3)


def rank_biserial(u_a, n_a, n_b):
    """Rank-biserial r from Mann-Whitney U of group A. Range -1..+1."""
    return (2.0 * u_a) / (n_a * n_b) - 1.0


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)          # enforce monotonicity
        adj[k] = running
    return adj


# ---------------------------------------------------------------- load sample
rows = read_csv(MASTER)
rows = [r for r in rows if (r.get("excluded") or "").strip().lower() not in TRUTHY]
rows.sort(key=lambda r: int(r["participant_id"][1:]))

A = [r for r in rows if r["condition"].strip().upper() == "A"]
B = [r for r in rows if r["condition"].strip().upper() == "B"]

report: list[str] = []


def say(line=""):
    print(line)
    report.append(line)


say("=" * 78)
say("FORMATIVE USER STUDY - ANALYSIS OUTPUT")
say("Per user_study/ANALYSIS_PLAN.md (frozen 2026-08-24, pre-inspection)")
say("=" * 78)
say()
say(f"Included: {len(A)} Condition A, {len(B)} Condition B, {len(rows)} total")
say("Unit of analysis: PARTICIPANT. No per-turn inferential statistics.")
say()

# ------------------------------------------------------- 1. questionnaire
say("-" * 78)
say("1. QUESTIONNAIRE Q1-Q11  (ordinal 1-5)")
say("-" * 78)
say("P09's immediate questionnaire is invalid and treated as MISSING (not replaced).")
say("Q5 is reverse-scored: higher = less interrupted, consistent with other items.")
say()

def valid_q(r):
    return (r.get("questionnaire_completed") or "").strip() in TRUTHY

qA = [r for r in A if valid_q(r)]
qB = [r for r in B if valid_q(r)]
say(f"Questionnaire n: Condition A = {len(qA)}, Condition B = {len(qB)}")
say()

hdr = (f"{'item':<5} {'nA':>3} {'nB':>3} {'medA':>5} {'medB':>5} "
       f"{'IQR A':>11} {'IQR B':>11} {'U':>6} {'r_rb':>6} {'p':>8} {'p_holm':>8}")
say(hdr)
say("-" * len(hdr))

raw_p, results = {}, {}
for q in LIKERT:
    a = [num(r.get(q)) for r in qA]
    b = [num(r.get(q)) for r in qB]
    if q == "Q5":                      # reverse-score: 6 - x
        a = [6 - v if v is not None else None for v in a]
        b = [6 - v if v is not None else None for v in b]
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if len(a) < 2 or len(b) < 2:
        say(f"{q:<5} insufficient data")
        continue
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    r_rb = rank_biserial(u, len(a), len(b))
    qa1, qa3 = iqr(a)
    qb1, qb3 = iqr(b)
    raw_p[q] = float(p)
    results[q] = dict(nA=len(a), nB=len(b), medA=st.median(a), medB=st.median(b),
                      iqrA=[qa1, qa3], iqrB=[qb1, qb3], U=float(u), r_rb=r_rb, p=float(p),
                      distA={k: a.count(k) for k in (1.0, 2.0, 3.0, 4.0, 5.0)},
                      distB={k: b.count(k) for k in (1.0, 2.0, 3.0, 4.0, 5.0)})

adj = holm(raw_p)
for q in LIKERT:
    if q not in results:
        continue
    d = results[q]
    d["p_holm"] = adj[q]
    say(f"{q:<5} {d['nA']:>3} {d['nB']:>3} {d['medA']:>5.1f} {d['medB']:>5.1f} "
        f"{str(d['iqrA']):>11} {str(d['iqrB']):>11} {d['U']:>6.1f} {d['r_rb']:>+6.2f} "
        f"{d['p']:>8.4f} {d['p_holm']:>8.4f}")

say()
say("r_rb = rank-biserial correlation. POSITIVE favours Condition A, negative favours B.")
say("Tests are EXPLORATORY. With n=9-10 per arm, read direction/magnitude, not p<.05.")
say()
say("Full 1-5 response distributions:")
for q in LIKERT:
    if q not in results:
        continue
    d = results[q]
    da = " ".join(f"{int(k)}:{d['distA'][k]}" for k in sorted(d["distA"]))
    db = " ".join(f"{int(k)}:{d['distB'][k]}" for k in sorted(d["distB"]))
    say(f"  {q:<4} A[{da}]   B[{db}]   {QUESTION_TEXT.get(q,'')[:44]}")

# --------------------------------------------- 2. objective, participant-level
say()
say("-" * 78)
say("2. OBJECTIVE METRICS  (aggregated per participant, then 10 vs 10)")
say("-" * 78)

def session_metrics(r):
    p = ROOT / r["raw_log"]
    srows = read_csv(p)
    if not srows:
        return None
    fto = [num(x.get("fto_ms")) for x in srows]
    fto = [v for v in fto if v is not None and v > 0]
    graph = [num(x.get("graph_latency_ms")) for x in srows]
    graph = [v for v in graph if v is not None and v > 0]
    ep = [num(x.get("endpoint_delay_ms")) for x in srows]
    ep = [v for v in ep if v is not None and v > 0]
    bot = [x for x in srows if (x.get("bot_text") or "").strip()]
    ts = [num(x.get("timestamp_ms")) for x in srows]
    ts = [v for v in ts if v]
    return dict(
        participant=r["participant_id"], condition=r["condition"],
        robot_turns=len(bot),
        fto_median=st.median(fto) if fto else None,
        fto_iqr=iqr(fto) if len(fto) > 1 else (None, None),
        graph_median=st.median(graph) if graph else None,
        endpoint_median=st.median(ep) if ep else None,
        duration_s=(max(ts) - min(ts)) / 1000.0 if len(ts) > 1 else None,
    )

per = [m for m in (session_metrics(r) for r in rows) if m]
with (OUT / "participant_level_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["participant", "condition", "robot_turns", "fto_median_ms",
                "fto_q1", "fto_q3", "graph_latency_median_ms",
                "endpoint_delay_median_ms", "duration_s"])
    for m in per:
        w.writerow([m["participant"], m["condition"], m["robot_turns"],
                    m["fto_median"], m["fto_iqr"][0], m["fto_iqr"][1],
                    m["graph_median"], m["endpoint_median"], m["duration_s"]])

COMPARABLE = [("fto_median", "First-turn offset (FTO), ms"),
              ("graph_median", "Backend graph latency, ms"),
              ("robot_turns", "Robot turns per session"),
              ("duration_s", "Session duration, s")]

say(f"{'metric':<34} {'medA':>9} {'medB':>9} {'U':>6} {'r_rb':>6} {'p':>8}")
say("-" * 78)
obj = {}
for key, label in COMPARABLE:
    a = [m[key] for m in per if m["condition"] == "A" and m[key] is not None]
    b = [m[key] for m in per if m["condition"] == "B" and m[key] is not None]
    if len(a) < 2 or len(b) < 2:
        continue
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    r_rb = rank_biserial(u, len(a), len(b))
    obj[key] = dict(label=label, nA=len(a), nB=len(b), medA=st.median(a), medB=st.median(b),
                    U=float(u), r_rb=r_rb, p=float(p))
    say(f"{label:<34} {st.median(a):>9.0f} {st.median(b):>9.0f} {u:>6.1f} {r_rb:>+6.2f} {p:>8.4f}")

say()
say("NOTE: endpoint_delay_ms exists only in Condition A logs - not comparable across arms.")
a_ep = [m["endpoint_median"] for m in per if m["condition"] == "A" and m["endpoint_median"]]
if a_ep:
    say(f"  Condition A endpoint delay, median of participant medians: {st.median(a_ep):.0f} ms")

# ------------------------------------------------------------ 3. qualitative
say()
say("-" * 78)
say("3. QUALITATIVE Q12-Q14  (available-case; NO prevalence comparison)")
say("-" * 78)

def kind(v):
    v = (v or "").strip()
    if not v:
        return "blank"
    if v.replace(".", "", 1).isdigit():
        return "NUMERIC"
    return "text"

with (OUT / "free_text_for_coding.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["participant", "condition", "question", "response"])
    for r in rows:
        for q in OPEN:
            if kind(r.get(q)) == "text":
                w.writerow([r["participant_id"], r["condition"], q, r[q].strip()])

for cond, grp in (("A", A), ("B", B)):
    usable = sum(1 for r in grp if any(kind(r.get(q)) == "text" for q in OPEN))
    say(f"Condition {cond}: {usable}/{len(grp)} participants gave at least one free-text answer")
say()
say("Missingness is CONDITION- AND PHASE-LINKED, not random: every numeric-instead-of-text")
say("failure (P01, P03, P09) is a 16 Aug Condition A session run BEFORE the prompt_multiline")
say("fix; all Condition B sessions ran after it. Report rates; do NOT compare theme frequencies.")

# ---------------------------------------------------------------- write files
(OUT / "questionnaire_results.json").write_text(
    json.dumps({"questionnaire": results, "objective": obj}, indent=2), encoding="utf-8")
(OUT / "analysis_report.txt").write_text("\n".join(report), encoding="utf-8")

say()
say("=" * 78)
say(f"Written to {OUT}:")
say("  analysis_report.txt            full text of this output")
say("  questionnaire_results.json     machine-readable results")
say("  participant_level_metrics.csv  one row per participant")
say("  free_text_for_coding.csv       Q12-Q14 responses ready for thematic coding")
say("=" * 78)
