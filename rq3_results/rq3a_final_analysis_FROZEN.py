from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    raise SystemExit(
        "SciPy is required. Install with: pip install scipy"
    )

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "Matplotlib is required. Install with: pip install matplotlib"
    )


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("rq3_results")

RAW_PATH = ROOT / (
    "rq3_raw_p100_final_rq3a_20260811_121928.csv"
)

RESULT_MANIFEST = ROOT / (
    "rq3a_final_result_manifest.json"
)

CONDITIONS = [
    "full",
    "no_safety",
    "no_persona",
]

EXPECTED_REPS = 24
EXPECTED_TURNS = 792


# ============================================================
# VERIFY FROZEN FINAL DATA
# ============================================================

def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


manifest = json.loads(
    RESULT_MANIFEST.read_text(
        encoding="utf-8"
    )
)

errors = []

for name, info in manifest["files"].items():

    p = Path(name)

    if not p.exists():
        errors.append(
            f"Missing: {p}"
        )
        continue

    current = sha256(p)

    if current != info["sha256"]:
        errors.append(
            f"Hash mismatch: {p}"
        )

if errors:

    print("FROZEN FINAL DATA CHECK: FAIL")

    for e in errors:
        print(e)

    raise SystemExit(1)

print(
    "FROZEN FINAL DATA CHECK: PASS"
)


# ============================================================
# LOAD RAW DATA
# ============================================================

df = pd.read_csv(RAW_PATH)

if len(df) != EXPECTED_TURNS:
    raise RuntimeError(
        f"Expected {EXPECTED_TURNS} rows, "
        f"found {len(df)}."
    )


required = [
    "repetition",
    "condition",
    "turn",
    "final_output_contract_pass",
    "final_strict_output_contract_pass",
    "raw_output_contract_pass",
    "safety_intervened",
    "topic_state_pass",
    "proper_closing",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# FIND LATENCY COLUMN
# ============================================================

latency_candidates = [
    "response_ready_ms",
    "response_ready_latency_ms",
    "latency_ms",
    "backend_response_ready_ms",
]

LATENCY_COL = None

for c in latency_candidates:
    if c in df.columns:
        LATENCY_COL = c
        break

if LATENCY_COL is None:

    possible = [
        c for c in df.columns
        if "latency" in c.lower()
        or "response_ready" in c.lower()
    ]

    if len(possible) == 1:
        LATENCY_COL = possible[0]

if LATENCY_COL is None:

    print("\nAVAILABLE COLUMNS:")
    for c in df.columns:
        print(" ", c)

    raise RuntimeError(
        "Could not identify latency column."
    )

print(
    "Latency column:",
    LATENCY_COL
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_cols = [
    "repetition",
    "turn",
    LATENCY_COL,
    "final_output_contract_pass",
    "final_strict_output_contract_pass",
    "raw_output_contract_pass",
    "safety_intervened",
    "topic_state_pass",
    "proper_closing",
]

for c in numeric_cols:
    df[c] = pd.to_numeric(
        df[c],
        errors="raise"
    )


# Optional raw strict metric
RAW_STRICT_COL = None

for c in [
    "raw_strict_output_contract_pass",
    "raw_strict_contract_pass",
]:
    if c in df.columns:
        RAW_STRICT_COL = c
        df[c] = pd.to_numeric(
            df[c],
            errors="raise"
        )
        break


# ============================================================
# SESSION-LEVEL AGGREGATION
# ============================================================

session = (
    df
    .groupby(
        ["repetition", "condition"],
        as_index=False
    )
    .agg(
        mean_latency_ms=(
            LATENCY_COL,
            "mean"
        ),
        median_latency_ms=(
            LATENCY_COL,
            "median"
        ),
        behavioral_pass_rate=(
            "final_output_contract_pass",
            "mean"
        ),
        strict_pass_rate=(
            "final_strict_output_contract_pass",
            "mean"
        ),
        raw_behavioral_pass_rate=(
            "raw_output_contract_pass",
            "mean"
        ),
        safety_intervention_rate=(
            "safety_intervened",
            "mean"
        ),
        topic_state_pass_rate=(
            "topic_state_pass",
            "mean"
        ),
    )
)


closing = (
    df[df["turn"] == 11]
    [["repetition", "condition", "proper_closing"]]
    .rename(
        columns={
            "proper_closing":
                "proper_closing_session"
        }
    )
)

session = session.merge(
    closing,
    on=["repetition", "condition"],
    how="left"
)

if RAW_STRICT_COL:

    raw_strict = (
        df
        .groupby(
            ["repetition", "condition"],
            as_index=False
        )[RAW_STRICT_COL]
        .mean()
        .rename(
            columns={
                RAW_STRICT_COL:
                    "raw_strict_pass_rate"
            }
        )
    )

    session = session.merge(
        raw_strict,
        on=["repetition", "condition"],
        how="left"
    )


expected_sessions = (
    EXPECTED_REPS * len(CONDITIONS)
)

if len(session) != expected_sessions:
    raise RuntimeError(
        f"Expected {expected_sessions} "
        f"session rows, got {len(session)}."
    )


session.to_csv(
    ROOT / "rq3a_final_session_metrics.csv",
    index=False
)


# ============================================================
# CONDITION DESCRIPTIVES
# ============================================================

desc_rows = []

for condition in CONDITIONS:

    d = df[
        df["condition"] == condition
    ]

    s = session[
        session["condition"] == condition
    ]

    row = {
        "condition": condition,

        "n_turns": len(d),
        "n_sessions": len(s),

        "latency_mean_ms":
            d[LATENCY_COL].mean(),

        "latency_median_ms":
            d[LATENCY_COL].median(),

        "latency_sd_ms":
            d[LATENCY_COL].std(ddof=1),

        "latency_p95_ms":
            d[LATENCY_COL].quantile(0.95),

        "session_mean_latency_ms":
            s["mean_latency_ms"].mean(),

        "session_mean_latency_sd_ms":
            s["mean_latency_ms"].std(ddof=1),

        "behavioral_pass_rate":
            d[
                "final_output_contract_pass"
            ].mean(),

        "strict_pass_rate":
            d[
                "final_strict_output_contract_pass"
            ].mean(),

        "raw_behavioral_pass_rate":
            d[
                "raw_output_contract_pass"
            ].mean(),

        "topic_state_pass_rate":
            d[
                "topic_state_pass"
            ].mean(),

        "closing_pass_rate":
            d.loc[
                d["turn"] == 11,
                "proper_closing"
            ].mean(),

        "safety_intervention_rate":
            d["safety_intervened"].mean(),

        "safety_intervention_count":
            int(
                d["safety_intervened"].sum()
            ),
    }

    if RAW_STRICT_COL:
        row["raw_strict_pass_rate"] = (
            d[RAW_STRICT_COL].mean()
        )

    desc_rows.append(row)


descriptives = pd.DataFrame(
    desc_rows
)

descriptives.to_csv(
    ROOT /
    "rq3a_final_condition_descriptives.csv",
    index=False
)


# ============================================================
# STATISTICAL HELPERS
# ============================================================

RNG = np.random.default_rng(
    20260811
)


def bootstrap_ci(
    values,
    n_boot=20000,
    alpha=0.05
):

    x = np.asarray(
        values,
        dtype=float
    )

    n = len(x)

    means = np.empty(
        n_boot,
        dtype=float
    )

    for i in range(n_boot):

        sample = RNG.choice(
            x,
            size=n,
            replace=True
        )

        means[i] = sample.mean()

    lo = np.quantile(
        means,
        alpha / 2
    )

    hi = np.quantile(
        means,
        1 - alpha / 2
    )

    return float(lo), float(hi)


def paired_vectors(metric, other):

    p = session.pivot(
        index="repetition",
        columns="condition",
        values=metric
    ).sort_index()

    if len(p) != EXPECTED_REPS:
        raise RuntimeError(
            f"{metric}: expected "
            f"{EXPECTED_REPS} pairs."
        )

    return (
        p["full"].to_numpy(
            dtype=float
        ),
        p[other].to_numpy(
            dtype=float
        )
    )


def safe_wilcoxon(a, b):

    diff = np.asarray(a) - np.asarray(b)

    nonzero = diff[
        ~np.isclose(diff, 0)
    ]

    if len(nonzero) == 0:
        return 0.0, 1.0

    result = stats.wilcoxon(
        a,
        b,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def sign_test(diff):

    diff = np.asarray(
        diff,
        dtype=float
    )

    nz = diff[
        ~np.isclose(diff, 0)
    ]

    if len(nz) == 0:
        return 0, 0, 1.0

    positive = int(
        (nz > 0).sum()
    )

    negative = int(
        (nz < 0).sum()
    )

    result = stats.binomtest(
        positive,
        n=positive + negative,
        p=0.5,
        alternative="two-sided"
    )

    return (
        positive,
        negative,
        float(result.pvalue),
    )


def holm_adjust(pvalues):

    p = np.asarray(
        pvalues,
        dtype=float
    )

    m = len(p)

    order = np.argsort(p)

    adjusted = np.empty(
        m,
        dtype=float
    )

    running = 0.0

    for rank, idx in enumerate(
        order,
        start=1
    ):

        value = (
            (m - rank + 1)
            * p[idx]
        )

        value = min(
            1.0,
            value
        )

        running = max(
            running,
            value
        )

        adjusted[idx] = running

    return adjusted


# ============================================================
# PLANNED CONTRASTS
# ============================================================

contrast_rows = []


# ------------------------------------------------------------
# LATENCY:
# full vs no_safety
# full vs no_persona
# ------------------------------------------------------------

for other in [
    "no_safety",
    "no_persona",
]:

    full, ablated = paired_vectors(
        "mean_latency_ms",
        other
    )

    diff = full - ablated

    mean_diff = diff.mean()

    sd_diff = diff.std(
        ddof=1
    )

    se = sd_diff / math.sqrt(
        len(diff)
    )

    if sd_diff > 0:

        critical = stats.t.ppf(
            0.975,
            df=len(diff) - 1
        )

        ci_low = (
            mean_diff
            - critical * se
        )

        ci_high = (
            mean_diff
            + critical * se
        )

        dz = mean_diff / sd_diff

    else:
        ci_low = mean_diff
        ci_high = mean_diff
        dz = 0.0

    ttest = stats.ttest_rel(
        full,
        ablated
    )

    w_stat, w_p = safe_wilcoxon(
        full,
        ablated
    )

    boot_low, boot_high = (
        bootstrap_ci(diff)
    )

    positive, negative, sign_p = (
        sign_test(diff)
    )

    contrast_rows.append({
        "outcome":
            "session_mean_latency_ms",

        "contrast":
            f"full_vs_{other}",

        "full_mean":
            full.mean(),

        "ablated_mean":
            ablated.mean(),

        "mean_difference":
            mean_diff,

        "difference_unit":
            "ms_full_minus_ablated",

        "relative_difference_percent":
            (
                mean_diff
                / ablated.mean()
                * 100
            ),

        "ci95_low":
            ci_low,

        "ci95_high":
            ci_high,

        "bootstrap_ci95_low":
            boot_low,

        "bootstrap_ci95_high":
            boot_high,

        "cohens_dz":
            dz,

        "paired_t":
            float(
                ttest.statistic
            ),

        "paired_t_p":
            float(
                ttest.pvalue
            ),

        "wilcoxon_W":
            w_stat,

        "wilcoxon_p":
            w_p,

        "sign_positive":
            positive,

        "sign_negative":
            negative,

        "sign_test_p":
            sign_p,

        "n_pairs":
            len(diff),
    })


# ------------------------------------------------------------
# CONTRACT OUTCOMES
# ------------------------------------------------------------

for metric in [
    "behavioral_pass_rate",
    "strict_pass_rate",
]:

    for other in [
        "no_safety",
        "no_persona",
    ]:

        full, ablated = (
            paired_vectors(
                metric,
                other
            )
        )

        diff = full - ablated

        w_stat, w_p = (
            safe_wilcoxon(
                full,
                ablated
            )
        )

        boot_low, boot_high = (
            bootstrap_ci(diff)
        )

        positive, negative, sign_p = (
            sign_test(diff)
        )

        contrast_rows.append({
            "outcome":
                metric,

            "contrast":
                f"full_vs_{other}",

            "full_mean":
                full.mean(),

            "ablated_mean":
                ablated.mean(),

            "mean_difference":
                diff.mean(),

            "difference_unit":
                "proportion_full_minus_ablated",

            "relative_difference_percent":
                np.nan,

            "ci95_low":
                np.nan,

            "ci95_high":
                np.nan,

            "bootstrap_ci95_low":
                boot_low,

            "bootstrap_ci95_high":
                boot_high,

            "cohens_dz":
                np.nan,

            "paired_t":
                np.nan,

            "paired_t_p":
                np.nan,

            "wilcoxon_W":
                w_stat,

            "wilcoxon_p":
                w_p,

            "sign_positive":
                positive,

            "sign_negative":
                negative,

            "sign_test_p":
                sign_p,

            "n_pairs":
                len(diff),
        })


contrasts = pd.DataFrame(
    contrast_rows
)


# Holm adjustment separately
# within each outcome.
contrasts[
    "wilcoxon_p_holm"
] = np.nan

contrasts[
    "paired_t_p_holm"
] = np.nan

for outcome in contrasts[
    "outcome"
].unique():

    mask = (
        contrasts["outcome"]
        == outcome
    )

    idx = contrasts.index[mask]

    contrasts.loc[
        idx,
        "wilcoxon_p_holm"
    ] = holm_adjust(
        contrasts.loc[
            idx,
            "wilcoxon_p"
        ].to_numpy()
    )

    tmask = (
        mask
        & contrasts[
            "paired_t_p"
        ].notna()
    )

    tidx = contrasts.index[
        tmask
    ]

    if len(tidx):

        contrasts.loc[
            tidx,
            "paired_t_p_holm"
        ] = holm_adjust(
            contrasts.loc[
                tidx,
                "paired_t_p"
            ].to_numpy()
        )


contrasts.to_csv(
    ROOT /
    "rq3a_final_planned_contrasts.csv",
    index=False
)


# ============================================================
# SAFETY REPAIR CONTRIBUTION
# ============================================================

repair_rows = []

for condition in [
    "full",
    "no_persona",
]:

    s = session[
        session["condition"]
        == condition
    ].copy()

    raw = s[
        "raw_behavioral_pass_rate"
    ].to_numpy()

    final = s[
        "behavioral_pass_rate"
    ].to_numpy()

    improvement = (
        final - raw
    )

    w_stat, w_p = safe_wilcoxon(
        final,
        raw
    )

    positive, negative, sign_p = (
        sign_test(improvement)
    )

    boot_low, boot_high = (
        bootstrap_ci(
            improvement
        )
    )

    turns = df[
        df["condition"]
        == condition
    ]

    repair_rows.append({
        "condition":
            condition,

        "raw_behavioral_pass_rate":
            turns[
                "raw_output_contract_pass"
            ].mean(),

        "final_behavioral_pass_rate":
            turns[
                "final_output_contract_pass"
            ].mean(),

        "raw_behavioral_failures":
            int(
                (
                    turns[
                        "raw_output_contract_pass"
                    ] == 0
                ).sum()
            ),

        "final_behavioral_failures":
            int(
                (
                    turns[
                        "final_output_contract_pass"
                    ] == 0
                ).sum()
            ),

        "safety_interventions":
            int(
                turns[
                    "safety_intervened"
                ].sum()
            ),

        "safety_intervention_rate":
            turns[
                "safety_intervened"
            ].mean(),

        "mean_session_improvement":
            improvement.mean(),

        "bootstrap_ci95_low":
            boot_low,

        "bootstrap_ci95_high":
            boot_high,

        "wilcoxon_W":
            w_stat,

        "wilcoxon_p":
            w_p,

        "sign_positive":
            positive,

        "sign_negative":
            negative,

        "sign_test_p":
            sign_p,
    })


repair = pd.DataFrame(
    repair_rows
)

repair.to_csv(
    ROOT /
    "rq3a_final_safety_repair_effect.csv",
    index=False
)


# ============================================================
# PLOTS
# ============================================================

plot_conditions = [
    "full",
    "no_safety",
    "no_persona",
]


# 1. Session mean latency
fig, ax = plt.subplots(
    figsize=(7, 5)
)

values = [
    session.loc[
        session["condition"] == c,
        "mean_latency_ms"
    ].to_numpy()
    for c in plot_conditions
]

ax.boxplot(
    values,
    tick_labels=plot_conditions
)

ax.set_ylabel(
    "Session mean response-ready latency (ms)"
)

ax.set_xlabel(
    "RQ3-A condition"
)

ax.set_title(
    "RQ3-A Backend Latency by Condition"
)

fig.tight_layout()

fig.savefig(
    ROOT /
    "rq3a_latency_session_boxplot.png",
    dpi=300
)

plt.close(fig)


# 2. Strict compliance
fig, ax = plt.subplots(
    figsize=(7, 5)
)

strict_values = [
    df.loc[
        df["condition"] == c,
        "final_strict_output_contract_pass"
    ].mean() * 100
    for c in plot_conditions
]

ax.bar(
    plot_conditions,
    strict_values
)

ax.set_ylim(
    0,
    105
)

ax.set_ylabel(
    "Strict output-contract compliance (%)"
)

ax.set_xlabel(
    "RQ3-A condition"
)

ax.set_title(
    "RQ3-A Strict Output-Contract Compliance"
)

fig.tight_layout()

fig.savefig(
    ROOT /
    "rq3a_strict_compliance.png",
    dpi=300
)

plt.close(fig)


# 3. Behavioral compliance
fig, ax = plt.subplots(
    figsize=(7, 5)
)

behavior_values = [
    df.loc[
        df["condition"] == c,
        "final_output_contract_pass"
    ].mean() * 100
    for c in plot_conditions
]

ax.bar(
    plot_conditions,
    behavior_values
)

ax.set_ylim(
    90,
    101
)

ax.set_ylabel(
    "Behavioral contract compliance (%)"
)

ax.set_xlabel(
    "RQ3-A condition"
)

ax.set_title(
    "RQ3-A Behavioral Reliability"
)

fig.tight_layout()

fig.savefig(
    ROOT /
    "rq3a_behavioral_compliance.png",
    dpi=300
)

plt.close(fig)


# 4. Safety intervention rate
fig, ax = plt.subplots(
    figsize=(6, 5)
)

safety_conditions = [
    "full",
    "no_persona",
]

intervention_values = [
    df.loc[
        df["condition"] == c,
        "safety_intervened"
    ].mean() * 100
    for c in safety_conditions
]

ax.bar(
    safety_conditions,
    intervention_values
)

ax.set_ylim(
    0,
    max(intervention_values) * 1.2
)

ax.set_ylabel(
    "Turns requiring Safety intervention (%)"
)

ax.set_xlabel(
    "Safety-enabled condition"
)

ax.set_title(
    "RQ3-A Safety Intervention Frequency"
)

fig.tight_layout()

fig.savefig(
    ROOT /
    "rq3a_safety_intervention_rate.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report_path = (
    ROOT /
    "rq3a_final_statistics.txt"
)

with report_path.open(
    "w",
    encoding="utf-8"
) as out:

    def write(text=""):
        print(text)
        out.write(
            str(text) + "\n"
        )

    write(
        "========================================"
    )
    write(
        "RQ3-A FINAL STATISTICAL ANALYSIS"
    )
    write(
        "========================================"
    )

    write()

    write(
        f"Raw turns: {len(df)}"
    )

    write(
        f"Matched repetitions per condition: "
        f"{EXPECTED_REPS}"
    )

    write(
        f"Inferential unit: matched session "
        f"(n={EXPECTED_REPS} per condition)"
    )

    write()

    write(
        "===== CONDITION DESCRIPTIVES ====="
    )

    for _, r in descriptives.iterrows():

        write()

        write(
            r["condition"]
        )

        write(
            f"  latency mean: "
            f"{r['latency_mean_ms']:.3f} ms"
        )

        write(
            f"  latency median: "
            f"{r['latency_median_ms']:.3f} ms"
        )

        write(
            f"  latency SD: "
            f"{r['latency_sd_ms']:.3f} ms"
        )

        write(
            f"  latency P95: "
            f"{r['latency_p95_ms']:.3f} ms"
        )

        write(
            f"  behavioral pass: "
            f"{r['behavioral_pass_rate']*100:.2f}%"
        )

        write(
            f"  strict pass: "
            f"{r['strict_pass_rate']*100:.2f}%"
        )

        write(
            f"  safety interventions: "
            f"{int(r['safety_intervention_count'])} "
            f"({r['safety_intervention_rate']*100:.2f}%)"
        )

        write(
            f"  topic/state pass: "
            f"{r['topic_state_pass_rate']*100:.2f}%"
        )

        write(
            f"  closing pass: "
            f"{r['closing_pass_rate']*100:.2f}%"
        )

    write()

    write(
        "===== PLANNED PAIRED CONTRASTS ====="
    )

    for _, r in contrasts.iterrows():

        write()

        write(
            f"{r['outcome']} | "
            f"{r['contrast']}"
        )

        write(
            f"  full mean: "
            f"{r['full_mean']:.6f}"
        )

        write(
            f"  ablated mean: "
            f"{r['ablated_mean']:.6f}"
        )

        write(
            f"  difference: "
            f"{r['mean_difference']:.6f}"
        )

        write(
            f"  bootstrap 95% CI: "
            f"[{r['bootstrap_ci95_low']:.6f}, "
            f"{r['bootstrap_ci95_high']:.6f}]"
        )

        if not pd.isna(
            r["paired_t_p"]
        ):

            write(
                f"  paired t: "
                f"{r['paired_t']:.6f}"
            )

            write(
                f"  paired t p: "
                f"{r['paired_t_p']:.8g}"
            )

            write(
                f"  paired t Holm p: "
                f"{r['paired_t_p_holm']:.8g}"
            )

            write(
                f"  Cohen dz: "
                f"{r['cohens_dz']:.6f}"
            )

        write(
            f"  Wilcoxon W: "
            f"{r['wilcoxon_W']:.6f}"
        )

        write(
            f"  Wilcoxon p: "
            f"{r['wilcoxon_p']:.8g}"
        )

        write(
            f"  Wilcoxon Holm p: "
            f"{r['wilcoxon_p_holm']:.8g}"
        )

        write(
            f"  sign test: "
            f"+{int(r['sign_positive'])} / "
            f"-{int(r['sign_negative'])}"
        )

        write(
            f"  sign-test p: "
            f"{r['sign_test_p']:.8g}"
        )

    write()

    write(
        "===== SAFETY REPAIR EFFECT ====="
    )

    for _, r in repair.iterrows():

        write()

        write(
            r["condition"]
        )

        write(
            f"  raw behavioral pass: "
            f"{r['raw_behavioral_pass_rate']*100:.2f}%"
        )

        write(
            f"  final behavioral pass: "
            f"{r['final_behavioral_pass_rate']*100:.2f}%"
        )

        write(
            f"  raw failures: "
            f"{int(r['raw_behavioral_failures'])}"
        )

        write(
            f"  final failures: "
            f"{int(r['final_behavioral_failures'])}"
        )

        write(
            f"  interventions: "
            f"{int(r['safety_interventions'])}"
        )

        write(
            f"  intervention rate: "
            f"{r['safety_intervention_rate']*100:.2f}%"
        )

        write(
            f"  Wilcoxon p: "
            f"{r['wilcoxon_p']:.8g}"
        )

        write(
            f"  sign-test p: "
            f"{r['sign_test_p']:.8g}"
        )

    write()

    write(
        "Analysis complete."
    )


print()
print(
    "===== OUTPUT FILES ====="
)

outputs = [
    "rq3a_final_session_metrics.csv",
    "rq3a_final_condition_descriptives.csv",
    "rq3a_final_planned_contrasts.csv",
    "rq3a_final_safety_repair_effect.csv",
    "rq3a_final_statistics.txt",
    "rq3a_latency_session_boxplot.png",
    "rq3a_strict_compliance.png",
    "rq3a_behavioral_compliance.png",
    "rq3a_safety_intervention_rate.png",
]

for name in outputs:
    print(
        ROOT / name
    )
