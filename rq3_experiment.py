import argparse
import csv
import hashlib
import itertools
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import config

import multi_agent.rq3_agents as AG
from multi_agent.rq3_graph import build_rq3_graph
from multi_agent.rq3_state import INTERVIEW_TOPICS

# Reuse the EXACT frozen RQ2 scripted candidate inputs and
# initial multi-agent state instead of copying/retyping them.
from rq2_experiment import CANDIDATE_INPUTS, initial_multi_state


CONDITIONS = (
    "full",
    "no_safety",
    "no_persona",
)

# All 6 possible execution orders.
# Cycling through these balances condition position and order.
EXECUTION_ORDERS = list(itertools.permutations(CONDITIONS))

SEP = "|||"


def percentile(values, q):
    if not values:
        return 0.0

    xs = sorted(float(x) for x in values)

    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo

    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def clean_for_scoring(text):
    """
    Remove separator pipes only for linguistic scoring.
    Separator correctness itself is measured separately.
    """
    return re.sub(r"\|+", " ", text or "").strip()


def separator_rule_pass(text):
    """
    Strict output separator contract:
    exactly one canonical ||| and it must be final.
    """
    t = (text or "").strip()
    return int(
        t.count(SEP) == 1
        and t.endswith(SEP)
    )


def score_text(text, expected_questions):
    original = (text or "").strip()
    clean = clean_for_scoring(original)

    nonempty = int(bool(clean))

    if clean:
        question_units = AG._question_units(clean)
        words = len(clean.split())
    else:
        question_units = 0
        words = 0

    question_pass = int(
        bool(clean)
        and question_units == expected_questions
    )

    length_pass = int(
        bool(clean)
        and words <= 30
    )

    # RQ2-compatible corrected behavioral contract:
    # non-empty + expected question units + <=30 words.
    behavioral_contract = int(
        nonempty
        and question_pass
        and length_pass
    )

    sep_pass = separator_rule_pass(original)

    # Stronger RQ3 contract also includes the delimiter rule.
    strict_contract = int(
        behavioral_contract
        and sep_pass
    )

    return {
        "nonempty_response": nonempty,
        "question_units": question_units,
        "expected_question_units": expected_questions,
        "question_rule_pass": question_pass,
        "word_count": words,
        "length_rule_pass": length_pass,
        "separator_rule_pass": sep_pass,
        "output_contract_pass": behavioral_contract,
        "strict_output_contract_pass": strict_contract,
    }


def verify_frozen_rq2():
    """
    Check that files hashed before RQ3 work have not changed.
    """
    manifest_path = Path(
        "rq3_results/rq3_frozen_rq2_manifest.json"
    )

    if not manifest_path.exists():
        print(
            "WARNING: frozen RQ2 manifest not found; "
            "hash verification skipped."
        )
        return

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    failures = []

    for rel, info in manifest.items():
        p = Path(rel)

        if not p.exists():
            failures.append(f"{rel}: FILE MISSING")
            continue

        current = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()

        if current != info["sha256"]:
            failures.append(
                f"{rel}: HASH CHANGED"
            )

    if failures:
        print("\nFROZEN RQ2 VERIFICATION FAILED:")
        for x in failures:
            print("  ", x)

        raise RuntimeError(
            "A frozen RQ2 file changed after the RQ3 manifest "
            "was created. Stop before running RQ3."
        )

    print(
        f"Frozen RQ2 verification: PASS "
        f"({len(manifest)} files unchanged)"
    )


def make_initial_state(condition):
    # Start from EXACT RQ2 state initialization.
    state = initial_multi_state()

    # Add only the RQ3 experimental condition.
    state["rq3_condition"] = condition

    return state


def run_condition(graph, condition, repetition, position, order_string):
    state = make_initial_state(condition)
    rows = []

    for turn, user_text in enumerate(CANDIDATE_INPUTS, 1):

        state["last_user_message"] = user_text
        state["history"].append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        old_safety_len = len(
            state.get("safety_log", [])
        )

        t0 = time.perf_counter()
        state = graph.invoke(state)
        latency_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        final_response = (
            state.get("response", "") or ""
        ).strip()

        agent = state.get(
            "next_agent",
            "unknown"
        )

        phase = state.get(
            "interview_phase",
            ""
        )

        topic = state.get(
            "current_topic",
            ""
        )

        # Experimental expectation is determined by turn,
        # NOT by whichever agent happened to run.
        #
        # Turns 1-10 should produce interview questions.
        # Turn 11 should close with zero questions.
        expected_questions = (
            0 if turn == 11 else 1
        )

        final_score = score_text(
            final_response,
            expected_questions,
        )

        safety_entries = state.get(
            "safety_log",
            []
        )

        safety_violations = []
        raw_response = final_response
        fixed_response = final_response
        safety_intervened = 0

        if len(safety_entries) > old_safety_len:

            latest = safety_entries[-1]

            safety_violations = latest.get(
                "violations",
                [],
            )

            if not isinstance(
                safety_violations,
                list,
            ):
                safety_violations = [
                    str(safety_violations)
                ]

            raw_response = (
                latest.get(
                    "original",
                    final_response,
                )
                or ""
            ).strip()

            fixed_response = (
                latest.get(
                    "fixed",
                    final_response,
                )
                or ""
            ).strip()

            safety_intervened = int(
                bool(safety_violations)
                or raw_response != fixed_response
            )

        raw_score = score_text(
            raw_response,
            expected_questions,
        )

        # ----------------------------
        # Topic/state progression
        # ----------------------------

        if turn <= len(INTERVIEW_TOPICS):

            expected_topic = (
                INTERVIEW_TOPICS[turn - 1]
            )

            topic_state_pass = int(
                topic == expected_topic
                and agent != "feedback"
                and phase != "closing"
            )

        else:

            expected_topic = "CLOSING"

            topic_state_pass = int(
                agent == "feedback"
                and phase == "closing"
            )

        proper_closing = int(
            turn == 11
            and final_score[
                "nonempty_response"
            ]
            and final_score[
                "question_units"
            ] == 0
            and final_score[
                "word_count"
            ] <= 30
            and agent == "feedback"
            and phase == "closing"
        )

        row = {
            "repetition": repetition,
            "position": position,
            "execution_order": order_string,
            "condition": condition,
            "turn": turn,

            "agent": agent,
            "phase": phase,
            "expected_topic": expected_topic,
            "current_topic": topic,
            "topic_state_pass": topic_state_pass,

            "user_text": user_text,

            "raw_response": raw_response,
            "final_response": final_response,

            "backend_response_ready_ms":
                round(latency_ms, 3),

            "safety_violations":
                "|".join(safety_violations),

            "safety_intervened":
                safety_intervened,

            "proper_closing":
                proper_closing,

            # Final/user-visible scoring
            **{
                f"final_{k}": v
                for k, v in final_score.items()
            },

            # Raw specialist scoring
            **{
                f"raw_{k}": v
                for k, v in raw_score.items()
            },
        }

        rows.append(row)

        # Preserve actual final robot response in conversation
        # history, as in RQ2.
        state["history"].append(
            {
                "role": "assistant",
                "content": final_response,
            }
        )

    return rows


def summarize_sessions(rows):
    grouped = {}

    for row in rows:
        key = (
            int(row["repetition"]),
            row["condition"],
        )
        grouped.setdefault(key, []).append(row)

    output = []

    for (rep, condition), group in sorted(
        grouped.items()
    ):

        lat = [
            float(x[
                "backend_response_ready_ms"
            ])
            for x in group
        ]

        closing_rows = [
            x for x in group
            if int(x["turn"]) == 11
        ]

        output.append({
            "repetition": rep,
            "condition": condition,

            "position": group[0]["position"],
            "execution_order":
                group[0]["execution_order"],

            "n_turns": len(group),

            "latency_mean_ms":
                round(statistics.mean(lat), 3),

            "latency_median_ms":
                round(statistics.median(lat), 3),

            "latency_sd_ms":
                round(
                    statistics.stdev(lat)
                    if len(lat) > 1
                    else 0.0,
                    3,
                ),

            "latency_p95_ms":
                round(
                    percentile(lat, 0.95),
                    3,
                ),

            "final_question_rule_pass_rate":
                round(
                    statistics.mean(
                        int(x[
                            "final_question_rule_pass"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "final_length_rule_pass_rate":
                round(
                    statistics.mean(
                        int(x[
                            "final_length_rule_pass"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "final_separator_rule_pass_rate":
                round(
                    statistics.mean(
                        int(x[
                            "final_separator_rule_pass"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "final_output_contract_pass_rate":
                round(
                    statistics.mean(
                        int(x[
                            "final_output_contract_pass"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "final_strict_contract_pass_rate":
                round(
                    statistics.mean(
                        int(x[
                            "final_strict_output_contract_pass"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "raw_output_contract_pass_rate":
                round(
                    statistics.mean(
                        int(x[
                            "raw_output_contract_pass"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "raw_strict_contract_pass_rate":
                round(
                    statistics.mean(
                        int(x[
                            "raw_strict_output_contract_pass"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "topic_state_pass_rate":
                round(
                    statistics.mean(
                        int(x[
                            "topic_state_pass"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "safety_intervention_rate":
                round(
                    statistics.mean(
                        int(x[
                            "safety_intervened"
                        ])
                        for x in group
                    ),
                    4,
                ),

            "proper_closing":
                int(
                    bool(closing_rows)
                    and closing_rows[0][
                        "proper_closing"
                    ]
                ),
        })

    return output


def summarize_conditions(session_rows):
    output = []

    for condition in CONDITIONS:

        group = [
            x for x in session_rows
            if x["condition"] == condition
        ]

        if not group:
            continue

        def mean_field(name):
            return statistics.mean(
                float(x[name])
                for x in group
            )

        output.append({
            "condition": condition,
            "n_sessions": len(group),

            "latency_mean_ms":
                round(
                    mean_field(
                        "latency_mean_ms"
                    ),
                    3,
                ),

            "output_contract_pass_rate":
                round(
                    mean_field(
                        "final_output_contract_pass_rate"
                    ),
                    4,
                ),

            "strict_contract_pass_rate":
                round(
                    mean_field(
                        "final_strict_contract_pass_rate"
                    ),
                    4,
                ),

            "raw_output_contract_pass_rate":
                round(
                    mean_field(
                        "raw_output_contract_pass_rate"
                    ),
                    4,
                ),

            "topic_state_pass_rate":
                round(
                    mean_field(
                        "topic_state_pass_rate"
                    ),
                    4,
                ),

            "proper_closing_rate":
                round(
                    mean_field(
                        "proper_closing"
                    ),
                    4,
                ),

            "safety_intervention_rate":
                round(
                    mean_field(
                        "safety_intervention_rate"
                    ),
                    4,
                ),
        })

    return output


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--label",
        default="rq3_smoke",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be >= 1"
        )

    print(
        "=== RQ3 COMPONENT ABLATION EXPERIMENT ==="
    )

    print("Model:", AG.MODEL)
    print(
        "Temperature:",
        getattr(
            config,
            "OPENAI_TEMPERATURE",
            0.5,
        ),
    )
    print("Max tokens: 100")
    print("Conditions:", CONDITIONS)
    print(
        "Scripted turns:",
        len(CANDIDATE_INPUTS),
    )
    print(
        "Interview topics:",
        len(INTERVIEW_TOPICS),
    )

    if AG.MODEL != "qwen2.5:7b":
        raise RuntimeError(
            f"Unexpected RQ3 model: {AG.MODEL}"
        )

    if len(CANDIDATE_INPUTS) != 11:
        raise RuntimeError(
            "Expected exactly 11 frozen RQ2 candidate turns"
        )

    if len(INTERVIEW_TOPICS) != 10:
        raise RuntimeError(
            "Expected exactly 10 interview topics"
        )

    verify_frozen_rq2()

    print(
        "\nBuilding three isolated RQ3 graphs..."
    )

    graphs = {
        condition:
            build_rq3_graph(condition)
        for condition in CONDITIONS
    }

    print(
        "Graphs built successfully."
    )

    print(
        "\nWarming persistent Qwen/Ollama client..."
    )

    AG.warmup()

    print("Warmup complete.")

    all_rows = []

    for rep in range(
        1,
        args.repeats + 1,
    ):

        order = EXECUTION_ORDERS[
            (rep - 1)
            % len(EXECUTION_ORDERS)
        ]

        order_string = ">".join(order)

        print(
            f"\nRepetition "
            f"{rep}/{args.repeats} "
            f"| order={order_string}"
        )

        for position, condition in enumerate(
            order,
            1,
        ):

            print(
                f"  {condition}...",
                end="",
                flush=True,
            )

            result = run_condition(
                graphs[condition],
                condition,
                rep,
                position,
                order_string,
            )

            all_rows.extend(result)

            mean_latency = statistics.mean(
                x[
                    "backend_response_ready_ms"
                ]
                for x in result
            )

            failures = sum(
                1
                for x in result
                if not x[
                    "final_output_contract_pass"
                ]
            )

            strict_failures = sum(
                1
                for x in result
                if not x[
                    "final_strict_output_contract_pass"
                ]
            )

            interventions = sum(
                int(x[
                    "safety_intervened"
                ])
                for x in result
            )

            print(
                f" mean={mean_latency:.1f}ms"
                f" | behavioral_fail={failures}"
                f" | strict_fail={strict_failures}"
                f" | safety_interventions={interventions}"
            )

    session_rows = summarize_sessions(
        all_rows
    )

    condition_rows = summarize_conditions(
        session_rows
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    out_dir = Path("rq3_results")
    out_dir.mkdir(exist_ok=True)

    raw_path = out_dir / (
        f"rq3_raw_{args.label}_{stamp}.csv"
    )

    session_path = out_dir / (
        f"rq3_session_summary_"
        f"{args.label}_{stamp}.csv"
    )

    condition_path = out_dir / (
        f"rq3_condition_summary_"
        f"{args.label}_{stamp}.csv"
    )

    metadata_path = out_dir / (
        f"rq3_metadata_"
        f"{args.label}_{stamp}.json"
    )

    write_csv(
        raw_path,
        all_rows,
    )

    write_csv(
        session_path,
        session_rows,
    )

    write_csv(
        condition_path,
        condition_rows,
    )

    metadata = {
        "label": args.label,
        "research_question":
            "RQ3 component contribution / ablation analysis",

        "conditions": list(CONDITIONS),

        "condition_definitions": {
            "full":
                "rule orchestrator + specialists + persona/style + safety",
            "no_safety":
                "same architecture but specialist output bypasses safety",
            "no_persona":
                "same full architecture but explicit persona/style wording removed",
        },

        "repetitions":
            args.repeats,

        "turns_per_condition_per_repeat":
            len(CANDIDATE_INPUTS),

        "model":
            AG.MODEL,

        "temperature":
            getattr(
                config,
                "OPENAI_TEMPERATURE",
                0.5,
            ),

        "max_tokens":
            100,

        "endpoint":
            "http://127.0.0.1:11435/v1",

        "runtime":
            "self-hosted GPU inference on university infrastructure",

        "topics":
            INTERVIEW_TOPICS,

        "candidate_inputs":
            CANDIDATE_INPUTS,

        "execution_order_method":
            "cycle through all six permutations of the three conditions",

        "latency_metric":
            "backend_response_ready_ms = elapsed wall-clock time from graph invocation to returned response; excludes ASR and TTS",

        "behavioral_contract":
            "non-empty response + expected question-unit count + <=30 words",

        "strict_contract":
            "behavioral contract + exactly one final ||| separator",

        "notes": [
            "RQ1 and RQ2 frozen; no RQ1/RQ2 experiment is rerun.",
            "Safety conditions retain raw specialist response and final repaired response.",
            "No-safety condition exposes specialist output directly.",
            "Persona ablation changes only explicit persona/style wording; task/output constraints remain.",
        ],
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== CONDITION SUMMARY ==="
    )

    for row in condition_rows:
        print(
            f"{row['condition']:12s}"
            f" latency={row['latency_mean_ms']:.1f}ms"
            f" behavioral={row['output_contract_pass_rate']:.3f}"
            f" strict={row['strict_contract_pass_rate']:.3f}"
            f" topic={row['topic_state_pass_rate']:.3f}"
            f" closing={row['proper_closing_rate']:.3f}"
            f" intervention={row['safety_intervention_rate']:.3f}"
        )

    print("\nSaved:")
    print(" ", raw_path)
    print(" ", session_path)
    print(" ", condition_path)
    print(" ", metadata_path)


if __name__ == "__main__":
    main()
