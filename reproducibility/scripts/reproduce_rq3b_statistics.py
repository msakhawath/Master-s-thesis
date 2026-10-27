"""
reproduce_rq3b_statistics.py

Post-submission reproduction script, created October 2026 from the preserved
original experimental outputs to reproduce the statistics reported for RQ3-B
(thesis Section 8.7.3, Table 8.3). This file was not part of the original frozen
experiment code; no standalone script was retained from the original analysis.

Input (original data, unchanged):
    rq3_results/rq3b_final_collection.zip   - the 40 trial CSVs + frozen manifest
Reference values (original output, unchanged):
    reproducibility/rq3b_original_outputs/rq3b_final_statistics.txt

Run from any folder, for example from the repository root:
    python reproducibility/scripts/reproduce_rq3b_statistics.py

Requires numpy and scipy (p-values are skipped with a message if scipy is absent).
Bootstrap confidence intervals depend on random resampling; the original resampling
seed was not recorded, so the last digits of the CIs may differ from the thesis.
"""
import csv
import difflib
import io
import math
import os
import re
import statistics as st
import sys
import zipfile

try:
    import numpy as np
except ImportError:
    np = None
try:
    from scipy import stats
except ImportError:
    stats = None

def _find_root():
    """Use the current folder if it holds rq3_results/, otherwise the repository
    root two levels above this script, so the script runs from any folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.getcwd(), os.path.normpath(os.path.join(here, "..", ".."))):
        if os.path.isfile(os.path.join(cand, "rq3_results", "rq3b_final_collection.zip")):
            return cand
    sys.exit("Cannot find rq3_results/rq3b_final_collection.zip; run from the repository root.")


ROOT = _find_root()
ZIP = os.path.join(ROOT, "rq3_results", "rq3b_final_collection.zip")
SENSITIVITY = [1] + list(range(13, 21))       # protocol-adherent subset, thesis 6.3.3
EXPECTED = {                                   # from rq3b_final_statistics.txt / Table 8.3
    "endpoint_delay_ms": dict(simple=2511.947, pred=777.579, diff=-1734.368, t=-25.003,
                              p_t=1.97932e-15, p_w=0.000131627, dz=-5.736),
    "fto_ms": dict(simple=4476.526, pred=2825.947, diff=-1650.579, t=-15.662,
                   p_t=6.24176e-12, p_w=3.8147e-06, dz=-3.593),
    "graph_latency_ms": dict(simple=1451.947, pred=1475.737, diff=23.789,
                             p_t=0.637031, p_w=0.198885, dz=0.110),
}


def norm(text):
    text = re.sub(r"[^a-z0-9 ]", "", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def p95(values):
    s = sorted(values)
    i = 0.95 * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def load_trials():
    trials, empty = {}, []
    with zipfile.ZipFile(ZIP) as z:
        for name in sorted(z.namelist()):
            m = re.search(r"FINAL_R(\d+)_", name)
            if not (m and name.endswith(".csv")):
                continue
            rep = int(m.group(1))
            cond = "simple" if "_simple_" in name else "pred"
            rows = list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8"))))
            if rows:
                trials[(rep, cond)] = rows[0]
            else:
                empty.append(name)
    return trials, empty


def compare(label, value, expected, fmt="{:.3f}"):
    flag = ""
    if expected is not None:
        tol = max(abs(expected) * 0.0005, 0.0015)
        flag = "  OK" if abs(value - expected) <= tol else f"  (thesis: {fmt.format(expected)})"
    print(f"    {label:34} {fmt.format(value)}{flag}")


def main():
    trials, empty = load_trials()
    reps = [r for r in range(1, 21) if (r, "simple") in trials and (r, "pred") in trials]
    print("RQ3-B REPRODUCTION (post-submission script)\n")
    print(f"Trial files with a logged row: {len(trials)} of 40")
    print(f"Empty trial files: {[os.path.basename(e) for e in empty]}")
    print(f"Complete matched pairs: {len(reps)}; excluded repetitions: "
          f"{[r for r in range(1, 21) if r not in reps]}")
    sims = [difflib.SequenceMatcher(None, norm(trials[(r, 'simple')]['user_text']),
                                    norm(trials[(r, 'pred')]['user_text'])).ratio() for r in reps]
    print(f"Minimum within-pair transcript similarity: {min(sims):.4f} (thesis: 0.9545)\n")

    pt, pw = {}, {}
    for metric, exp in EXPECTED.items():
        a = [float(trials[(r, "simple")][metric]) for r in reps]
        b = [float(trials[(r, "pred")][metric]) for r in reps]
        d = [y - x for x, y in zip(a, b)]
        md, sd = st.mean(d), st.stdev(d)
        print(f"== {metric}  (n = {len(d)} pairs)")
        compare("fixed 2.5 s, mean", st.mean(a), exp["simple"])
        print(f"    {'fixed 2.5 s, median / SD / P95':34} {st.median(a):.1f} / {st.stdev(a):.3f} / {p95(a):.1f}")
        compare("VAP+TurnGPT, mean", st.mean(b), exp["pred"])
        print(f"    {'VAP+TurnGPT, median / SD / P95':34} {st.median(b):.1f} / {st.stdev(b):.3f} / {p95(b):.1f}")
        compare("difference (pred - fixed)", md, exp["diff"])
        print(f"    {'percent change':34} {100 * md / st.mean(a):.2f}%")
        t = md / (sd / math.sqrt(len(d)))
        if "t" in exp:
            compare(f"paired t({len(d) - 1})", t, exp["t"])
        compare("Cohen dz", md / sd, exp["dz"])
        print(f"    {'pairs favouring VAP+TurnGPT':34} {sum(x < 0 for x in d)}/{len(d)}")
        if stats is not None:
            pt[metric] = stats.ttest_rel(b, a).pvalue
            pw[metric] = stats.wilcoxon(b, a).pvalue
            compare("paired t-test p", pt[metric], exp["p_t"], "{:.6g}")
            compare("Wilcoxon signed-rank p", pw[metric], exp["p_w"], "{:.6g}")
            k = sum(x < 0 for x in d)
            print(f"    {'sign test p':34} {stats.binomtest(k, len(d)).pvalue:.6g}")
        if np is not None:
            rng = np.random.default_rng(20260811)
            arr = np.array(d)
            boots = rng.choice(arr, size=(10000, len(arr)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"    {'bootstrap 95% CI (10,000 resamples)':34} [{lo:.3f}, {hi:.3f}]")
        print()

    if stats is not None:
        names = list(pt)
        for label, pv in (("paired t", pt), ("Wilcoxon", pw)):
            adj = holm([pv[n] for n in names])
            print(f"Holm-adjusted {label} p (3 outcomes): " +
                  ", ".join(f"{n}={a:.6g}" for n, a in zip(names, adj)))
        print()

    sens = [r for r in reps if r in SENSITIVITY]
    print(f"== Protocol-adherent sensitivity subset (n = {len(sens)}: R01, R13-R20)")
    for metric in ("endpoint_delay_ms", "fto_ms"):
        d = [float(trials[(r, "pred")][metric]) - float(trials[(r, "simple")][metric]) for r in sens]
        line = f"    {metric:20} mean difference {st.mean(d):.3f} ms"
        if stats is not None:
            line += f"; paired t p={stats.ttest_1samp(d, 0).pvalue:.6g}; Wilcoxon p={stats.wilcoxon(d).pvalue:.6g}"
        print(line)

    pred_all = [row for (r, c), row in trials.items() if c == "pred"]
    print(f"\n== Predictive-controller diagnostics (all {len(pred_all)} predictive trials)")
    for col in ("yield_reason", "turngpt_timeout_ms", "vap_backend"):
        counts = {}
        for row in pred_all:
            counts[row[col]] = counts.get(row[col], 0) + 1
        print(f"    {col:22} {counts}")
    raw = [float(r["turngpt_raw_probability"]) for r in pred_all]
    shaped = [float(r["turngpt_shaped_probability"]) for r in pred_all]
    print(f"    mean TurnGPT raw / shaped probability: {st.mean(raw):.6f} / {st.mean(shaped):.6f}")
    if stats is None:
        print("\nscipy not installed: p-values were skipped.")


if __name__ == "__main__":
    sys.exit(main())
