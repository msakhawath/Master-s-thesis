
import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import config
import llm_module_ollama as BASE
from multi_agent import agents as AG
from multi_agent.graph import interview_graph
from multi_agent.state import INTERVIEW_TOPICS


# Same controlled candidate utterances for BOTH architectures.
# Turns 1-10 lead to the ten required topics.
# Turn 11 answers career goals, after which the system should close.
CANDIDATE_INPUTS = [
    "Yes, I am ready.",
    "I completed a master's in Data Science and worked on several academic projects.",
    "I mainly use Python, SQL, pandas, NumPy and scikit-learn.",
    "I have implemented classification, anomaly detection and machine learning models.",
    "I clean missing values, remove duplicates and inspect inconsistent observations.",
    "I evaluate models using precision, recall, F1, ROC AUC and cross-validation.",
    "I create visualizations with matplotlib and Power BI and explain results to stakeholders.",
    "I completed real-world projects involving preprocessing, modelling and evaluation.",
    "I collaborate with technical and non-technical colleagues and communicate findings clearly.",
    "I normally define the problem, inspect the data, test alternatives and validate the result.",
    "My goal is to become a strong applied data scientist and work on useful real-world systems.",
]


def clean_text(text):
    return (text or "").replace("|||", "").strip()


def percentile(values, p):
    if not values:
        return None
    x = sorted(values)
    if len(x) == 1:
        return x[0]
    pos = (len(x) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(x) - 1)
    frac = pos - lo
    return x[lo] + frac * (x[hi] - x[lo])


def expected_for_turn(turn):
    if turn <= len(INTERVIEW_TOPICS):
        return INTERVIEW_TOPICS[turn - 1], 1
    return "closing", 0


def make_row(
    architecture,
    repetition,
    turn,
    user_text,
    response,
    latency_ms,
    agent="",
    actual_topic="",
    safety_violations="",
    condition_position=1,
):
    expected_topic, expected_questions = expected_for_turn(turn)

    clean = clean_text(response)
    question_count = clean.count("?")
    word_count = len(clean.split())

    return {
        "architecture": architecture,
        "repetition": repetition,
        "condition_position": condition_position,
        "turn": turn,
        "expected_topic": expected_topic,
        "actual_topic": actual_topic,
        "agent": agent,
        "user_text": user_text,
        "response": clean,
        "backend_response_ready_ms": round(latency_ms, 3),
        "question_count": question_count,
        "expected_question_count": expected_questions,
        "question_rule_pass": int(question_count == expected_questions),
        "word_count": word_count,
        "under_30_words": int(word_count <= 30),
        "output_contract_pass": int(
            question_count == expected_questions and word_count <= 30
        ),
        "safety_violations": safety_violations,
    }


def run_baseline(repetition, position):
    history = []
    rows = []

    for turn, user_text in enumerate(CANDIDATE_INPUTS, 1):
        history.append({"role": "user", "content": user_text})

        t0 = time.perf_counter()
        chunks = list(BASE.llm_stream(history))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        response = " ".join(
            chunk for chunk, _ in chunks if chunk
        ).strip()

        rows.append(
            make_row(
                architecture="baseline",
                repetition=repetition,
                turn=turn,
                user_text=user_text,
                response=response,
                latency_ms=latency_ms,
                agent="single_agent",
                condition_position=position,
            )
        )

        history.append({"role": "assistant", "content": response})

    return rows


def initial_multi_state():
    return {
        "history": [],
        "last_user_message": "",
        "current_topic": INTERVIEW_TOPICS[0],
        "topics_covered": [],
        "probed_current_topic": False,
        "interview_phase": "opening",
        "next_agent": "interviewer",
        "response": "",
        "safety_log": [],
    }


def run_multi_agent(repetition, position):
    state = initial_multi_state()
    rows = []

    for turn, user_text in enumerate(CANDIDATE_INPUTS, 1):
        state["last_user_message"] = user_text
        state["history"].append(
            {"role": "user", "content": user_text}
        )

        old_safety_len = len(state.get("safety_log", []))

        t0 = time.perf_counter()
        state = interview_graph.invoke(state)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        response = state.get("response", "")

        safety_entries = state.get("safety_log", [])
        violations = ""

        if len(safety_entries) > old_safety_len:
            latest = safety_entries[-1]
            v = latest.get("violations", [])
            if isinstance(v, list):
                violations = "|".join(v)
            else:
                violations = str(v or "")

        rows.append(
            make_row(
                architecture="multi_agent",
                repetition=repetition,
                turn=turn,
                user_text=user_text,
                response=response,
                latency_ms=latency_ms,
                agent=state.get("next_agent", ""),
                actual_topic=state.get("current_topic", ""),
                safety_violations=violations,
                condition_position=position,
            )
        )

        state["history"].append(
            {"role": "assistant", "content": response}
        )

    return rows


def summarize(rows):
    output = []

    for arch in ("baseline", "multi_agent"):
        r = [x for x in rows if x["architecture"] == arch]
        lat = [float(x["backend_response_ready_ms"]) for x in r]

        question_pass = sum(int(x["question_rule_pass"]) for x in r)
        contract_pass = sum(int(x["output_contract_pass"]) for x in r)

        closing = [x for x in r if int(x["turn"]) == 11]
        closing_pass = (
            sum(int(x["question_rule_pass"]) for x in closing) / len(closing)
            if closing else 0
        )

        safety_turns = sum(
            1 for x in r if str(x["safety_violations"]).strip()
        )

        output.append({
            "architecture": arch,
            "n_turns": len(r),
            "latency_mean_ms": round(statistics.mean(lat), 3),
            "latency_median_ms": round(statistics.median(lat), 3),
            "latency_sd_ms": round(
                statistics.stdev(lat) if len(lat) > 1 else 0, 3
            ),
            "latency_p95_ms": round(percentile(lat, 0.95), 3),
            "question_rule_pass_rate": round(question_pass / len(r), 4),
            "output_contract_pass_rate": round(contract_pass / len(r), 4),
            "closing_pass_rate": round(closing_pass, 4),
            "safety_violation_turns": safety_turns,
        })

    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--label", default="p100_final_rq2")
    args = parser.parse_args()

    print("=== RQ2 CONTROLLED EXPERIMENT ===")
    print("Baseline model:", BASE.OLLAMA_MODEL)
    print("Multi-agent model:", AG.MODEL)
    print("Baseline controlled prompt:", not BASE.USE_FULL_PROMPT)
    print("Repetitions:", args.repeats)

    if BASE.OLLAMA_MODEL != AG.MODEL:
        raise RuntimeError("MODEL MISMATCH between architectures")

    print("\nWarming persistent clients...")
    BASE.warmup()
    AG.warmup()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("rq2_results")
    out_dir.mkdir(exist_ok=True)

    all_rows = []

    for rep in range(1, args.repeats + 1):
        # Alternate execution order to reduce systematic drift/order bias.
        if rep % 2 == 1:
            order = [
                ("baseline", run_baseline),
                ("multi_agent", run_multi_agent),
            ]
        else:
            order = [
                ("multi_agent", run_multi_agent),
                ("baseline", run_baseline),
            ]

        print(f"\nRepetition {rep}/{args.repeats}")

        for position, (name, fn) in enumerate(order, 1):
            print(f"  {name}...", end="", flush=True)
            result = fn(rep, position)
            all_rows.extend(result)

            mean_lat = statistics.mean(
                x["backend_response_ready_ms"] for x in result
            )
            failures = sum(
                1 for x in result if not x["output_contract_pass"]
            )

            print(
                f" mean={mean_lat:.1f}ms "
                f"contract_failures={failures}"
            )

    raw_path = out_dir / f"rq2_raw_{args.label}_{stamp}.csv"
    write_csv(raw_path, all_rows)

    summary = summarize(all_rows)
    summary_path = out_dir / f"rq2_summary_{args.label}_{stamp}.csv"
    write_csv(summary_path, summary)

    metadata = {
        "label": args.label,
        "repetitions": args.repeats,
        "turns_per_condition_per_repeat": len(CANDIDATE_INPUTS),
        "model": BASE.OLLAMA_MODEL,
        "temperature": getattr(config, "OPENAI_TEMPERATURE", 0.5),
        "max_tokens": 100,
        "baseline_prompt_mode": "controlled_monolithic_single_agent",
        "multi_agent": "rule_orchestrator_plus_specialists_plus_safety",
        "endpoint": "http://127.0.0.1:11435/v1",
        "topics": INTERVIEW_TOPICS,
        "metric_definition":
            "backend_response_ready_ms = elapsed wall-clock time from "
            "architecture invocation to usable returned text; excludes ASR/TTS.",
    }

    metadata_path = out_dir / f"rq2_metadata_{args.label}_{stamp}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    for row in summary:
        print(row)

    print("\nSaved:")
    print(raw_path)
    print(summary_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
