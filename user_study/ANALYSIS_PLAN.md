# Human Study Analysis Plan — FROZEN BEFORE INFERENTIAL INSPECTION

**Frozen: 2026-08-24, before any between-condition comparison was computed or viewed.**

This document is the pre-specification for the formative user evaluation. It exists so that
every analytic choice can be shown to predate sight of the group results. Nothing in it may be
revised in the light of an outcome.

---

## 0. The binding rule

> **No new outcome definition, subgroup, transformation, exclusion or test may be introduced
> after inspecting condition differences, unless it is explicitly labelled EXPLORATORY in the
> thesis.**

Anything added later is reported as exploratory, in its own subsection, and never as a
confirmatory result.

---

## 1. Sample and unit of analysis

| | |
|---|---|
| Unit of analysis | **participant / session** — never the individual turn |
| Included sample | 10 Condition A, 10 Condition B (P01–P20) |
| Excluded participants | none |
| Allocation | systematic alternating assignment, odd ID → A, even ID → B (`ALLOCATION_AMENDMENT.md`) |

**Pseudo-replication is prohibited.** Objective metrics are aggregated to one value per
participant first; the comparison is then 10 summaries against 10. Per-turn data may be used
for plots and diagnostic explanation only, never for inferential tests.

---

## 2. Questionnaire (Q1–Q11, 5-point ordinal)

| | |
|---|---|
| Valid n | reported **per item and per condition** |
| P09 | immediate questionnaire invalid (documented). Values treated as **missing**, not replaced. Condition A questionnaire n = **9**; P09 remains in all objective analyses (n = 10) |
| Primary descriptives | **median and IQR** |
| Distribution | full 1–5 response counts per item per condition |
| Mean / SD | supplementary only |
| Q5 | reverse-scored consistently; raw Q5 retained alongside |
| Test | Mann–Whitney U, **exploratory** |
| Effect size | rank-biserial *r* — **emphasised over p-values** |
| p-values | raw **and** Holm-adjusted across the 11 items, both reported |
| CIs | reported where the method is documented and reproducible |

**No composite score.** Q1–Q11 measure different aspects of turn-taking and interaction
quality; they are not summed or averaged unless a specific aggregation is justified here in
advance, and none is.

**Framing.** With 9–10 observations per arm this is a formative evaluation. The thesis
foregrounds direction, magnitude, distribution and consistency with the objective logs.
"Significant / non-significant" is not the headline, and per-item p-values are unstable at this
n by construction.

---

## 3. Objective interaction metrics

One row per participant, then 10 vs 10:

- median FTO · IQR FTO · (p90 if useful)
- median endpoint-related delay
- number of robot turns
- session duration
- technical incidents · researcher interventions
- premature-interruption count (**Condition A only** — see §4)

---

## 4. Premature interruption — Condition A only

**The two arms log different, non-overlapping indicators.** Condition A has
`user_start_ms` / `acoustic_stop_ms`; Condition B does not. Condition B's `interrupt_flag` is
`False` on all 1660 rows because barge-in is disabled, and `false_cutin_flag`,
`barge_in_stop` and `overlap_prev_turn` are empty. **No comparable log-derived rate exists.**

Therefore:

- **Between-condition interruption claim rests on Q5**, the only measure present in both arms.
- The log measure is **within-Condition-A explanatory evidence** — it explains why Condition A
  participants may report interruption. It may **not** support "A interrupts more than B".

**Considered and DECLINED, pre-inspection (2026-08-24).** A supplementary measure — counting
turns whose transcript ends grammatically incomplete — was proposed. It would have been
computable in both arms and would have supplied a between-condition cut-off comparison. It was
**deliberately declined** in favour of a simpler design, **before any A-vs-B comparison was
computed or viewed**. Q5 alone carries the between-condition interruption claim. This is
recorded so the decision is auditable as pre-specified rather than outcome-driven.

**Known blind spot, stated in advance.** The candidate detector finds only **gate-dropout**
cut-offs. **Policy-timing** cut-offs — where the gate tracked the voice correctly and the
endpoint simply expired during a thinking pause — are invisible to it. P17 turn 7
(*"yes i can tell you but"*, 3397 ms of correctly detected speech) is one such case and was not
flagged. **The 2.7% figure is therefore a lower bound on cut-offs**, and must be reported as a
gate-dropout rate, not as a total interruption rate.

**Reporting decision (simplification, pre-inspection).** All four validated candidates are
reported as a single category — *gate-dropout events, 4/148 Condition A turns (2.7%)* — with no
sub-classification into "resumed" versus "abandoned" continuations. Only one (P15 turn 10)
shows the participant resuming the same sentence; the other three end mid-phrase without
resumption. That detail is recorded here but not used as a separate outcome.

**Candidate detector, then manual validation.** The impossible-speaking-rate test
(≥3 words transcribed within a detected speech window implying >5 words/second) establishes
that the RMS gate lost the voice — **not** that the robot spoke over a continuing answer.

- Candidates: **4 of 148 Condition A turns (2.7%)** — P13 turn 9, P15 turn 10, P17 turns 4 and 8.
- Each is validated by reading whether the following turn continues the same sentence.
- Only validated turns are reported as premature interruptions. The 2.7% arm-level rate is
  quoted in preference to per-session rates (P17 12%, P15 7%), which overstate the phenomenon.

---

## 5. Qualitative (Q12–Q14)

**Available-case thematic analysis. No between-condition prevalence inference.**

| | usable free-text |
|---|---|
| Condition A | 4 / 10 |
| Condition B | 9 / 10 |

**The missingness is condition- and phase-linked, not random.** Every numeric-instead-of-text
failure (P01, P03, P09) is a 16 August Condition A session run *before* the `prompt_multiline`
entry fix; every Condition B session ran after it. Condition A's four usable responses are
therefore not a random sample of Condition A. Post-fix blanks (P18, P19) are deliberate skips,
which the consent form permits.

Permitted: *"Several respondents described…"*, *"Among participants providing free-text
feedback…"*. **Prohibited:** any statement of the form "X% of B versus Y% of A mentioned…".
Themes are illustrative and explanatory, not a balanced comparison.

---

## 6. Limitations fixed in advance

- **Condition × collection time is heavily confounded.** 6 of 10 Condition A ran on 16 Aug, 1 on
  20 Aug, 3 on 23 Aug; all 10 Condition B ran on 23–24 Aug. Only three A sessions were
  interleaved with B. Study-day effects — researcher familiarity, audio configuration,
  recruitment order, participant composition — are not separable from condition.
- **Audio configuration drifted across collection days and was subsequently re-standardised.**
  Do **not** write "original → corrected": the 16 August sessions had **zero** gate dropouts and
  were the cleanest in the study. Drift occurred later and was re-standardised on 23 August.
- **Session concentration.** Twelve sessions on 23 August, two after midnight, all but three in
  Condition B. This may have introduced researcher/session-order effects, including fatigue.
  Fatigue was not measured and is not claimed as a finding — it is one reason the time/order
  confound matters.
- **Repeat exposure.** P02, P13 and P15 completed an earlier full interview before their rated
  session; their ratings are not first-encounter.
- **System-level comparison.** The arms differ in model, architecture, TTS, robot interface and
  controller. Never attribute a rating difference to the language model alone.

---

## 7. Reproducibility

The analysis pipeline is **read-only** with respect to study data. It writes exclusively to a
separate `analysis/results/` directory and modifies no file under `user_study/data/`,
`rq3_results/` or `logs/`. Study code remains frozen (`freeze_study_code.py --verify` PASS).
