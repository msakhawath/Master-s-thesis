from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
import asr_module as ASR
import furhat_realtime_module as FR
import vap_module as VAP
import turngpt_module as TG

from multi_agent.graph import interview_graph
from multi_agent.state import InterviewState, INTERVIEW_TOPICS
from multi_agent.agents import MODEL, warmup
from rq3b_policy import (
    simple_endpoint_decision,
    predictive_endpoint_decision,
)

TIME_LIMIT_S = float(getattr(config, "SESSION_TIME_LIMIT_S", 600.0))

GREETING = (
    "Hi, I am Furhat, your interviewer at DataTech Berlin. "
    "We are interviewing for a Junior Data Scientist position. "
    "Are you ready to begin?"
)

_start_time: Optional[float] = None


def now_ms() -> int:
    return int(time.time() * 1000)


def elapsed_s() -> float:
    return 0.0 if _start_time is None else time.time() - _start_time


def clean_for_speech(text: str) -> str:
    text = re.sub(r"\|+", " ", text or "")
    text = text.replace('"', "").replace("'", "")
    return re.sub(r"\s{2,}", " ", text).strip()


@dataclass
class EndpointResult:
    user_text: str
    user_start_ms: Optional[int]
    acoustic_stop_ms: int
    asr_final_ms: int
    endpoint_ms: int
    endpoint_delay_ms: int
    reason: str

    vap_p_now_user: Optional[float] = None
    vap_p_fut_user: Optional[float] = None
    vap_p_now_robot: Optional[float] = None
    vap_p_fut_robot: Optional[float] = None
    vap_forward_ms: Optional[float] = None
    vap_backend: str = ""

    turngpt_raw_probability: Optional[float] = None
    turngpt_shaped_probability: Optional[float] = None
    turngpt_prompt_tokens: Optional[int] = None
    turngpt_timeout_ms: Optional[int] = None


class RQ3BLogger:
    FIELDNAMES = [
        "timestamp_ms",
        "participant_id",
        "condition",
        "turn_index",
        "user_text",
        "bot_text",
        "agent",
        "topic",
        "yield_reason",
        "user_start_ms",
        "acoustic_stop_ms",
        "asr_final_ms",
        "asr_final_latency_ms",
        "endpoint_permission_ms",
        "endpoint_delay_ms",
        "graph_latency_ms",
        "robot_start_ms",
        "fto_ms",
        "furhat_gen_time_ms",
        "vap_p_now_user",
        "vap_p_fut_user",
        "vap_p_now_robot",
        "vap_p_fut_robot",
        "vap_forward_ms",
        "vap_backend",
        "turngpt_raw_probability",
        "turngpt_shaped_probability",
        "turngpt_prompt_tokens",
        "turngpt_timeout_ms",
        "safety_violation_count",
    ]

    def __init__(self, condition: str, participant_id: str):
        out_dir = Path("rq3_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = out_dir / f"rq3b_{condition}_{participant_id}_{stamp}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()

    def write(self, row: dict) -> None:
        clean = {k: row.get(k, "") for k in self.FIELDNAMES}
        clean["timestamp_ms"] = now_ms()
        self.writer.writerow(clean)
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


def acoustic_speaking() -> bool:
    return bool(ASR.user_speaking())


def listen_until_endpoint(
    condition: str,
    state: InterviewState,
) -> Optional[EndpointResult]:
    final_text = ""
    last_partial_ms: Optional[int] = None

    user_start_ms: Optional[int] = None
    silence_start_ms: Optional[int] = None
    acoustic_stop_ms: Optional[int] = None
    asr_final_ms: Optional[int] = None

    last_vap = {}
    last_tg = {}

    while elapsed_s() < TIME_LIMIT_S:
        now = now_ms()

        # Drain all currently available ASR messages.
        while ASR.has_new_result():
            result = ASR.get_result()
            if not result:
                break

            kind, text = result
            text = (text or "").strip()

            if kind == "partial" and text:
                last_partial_ms = now
                if user_start_ms is None:
                    user_start_ms = now

            elif kind == "final" and text:
                final_text = f"{final_text} {text}".strip()
                asr_final_ms = now
                if user_start_ms is None:
                    user_start_ms = now

        # SAME acoustic anchor in both conditions.
        speaking = acoustic_speaking()

        if speaking:
            if user_start_ms is None:
                user_start_ms = now
            silence_start_ms = None
            acoustic_stop_ms = None
        else:
            if user_start_ms is not None and silence_start_ms is None:
                silence_start_ms = now
                acoustic_stop_ms = now

        if acoustic_stop_ms is None:
            time.sleep(0.02)
            continue

        gap_ms = max(0, now - acoustic_stop_ms)
        final_ready = bool(final_text and asr_final_ms is not None)

        if condition == "simple":
            decision = simple_endpoint_decision(
                final_ready=final_ready,
                user_speaking=speaking,
                gap_ms=gap_ms,
            )

        else:
            # Predictive condition only: VAP + TurnGPT.
            v = VAP.get_current(
                window_sec=float(getattr(config, "VAP_IDLE_WIN", 0.18))
            )
            last_vap = v

            tg_details = {}
            tg_score = 0.0

            if final_ready:
                temp_history = list(state["history"]) + [
                    {"role": "user", "content": final_text}
                ]
                tg_details = TG.get_turngpt_details(
                    temp_history,
                    partial=None,
                    for_prepare=False,
                )
                tg_score = float(tg_details["score"])
                last_tg = tg_details

            decision = predictive_endpoint_decision(
                final_ready=final_ready,
                user_speaking=bool(v.get("user_is_speaking", False)),
                gap_ms=gap_ms,
                p_now_user=float(v.get("p_now_user", 0.5)),
                p_future_user=float(v.get("p_fut_user", 0.5)),
                p_now_robot=float(v.get("p_now_robot", 0.5)),
                p_future_robot=float(v.get("p_fut_robot", 0.5)),
                tg_score=tg_score,
            )

        if decision.should_yield:
            endpoint_ms = now_ms()

            return EndpointResult(
                user_text=final_text,
                user_start_ms=user_start_ms,
                acoustic_stop_ms=acoustic_stop_ms,
                asr_final_ms=int(asr_final_ms),
                endpoint_ms=endpoint_ms,
                endpoint_delay_ms=endpoint_ms - acoustic_stop_ms,
                reason=decision.reason,
                vap_p_now_user=last_vap.get("p_now_user"),
                vap_p_fut_user=last_vap.get("p_fut_user"),
                vap_p_now_robot=last_vap.get("p_now_robot"),
                vap_p_fut_robot=last_vap.get("p_fut_robot"),
                vap_forward_ms=last_vap.get("forward_ms"),
                vap_backend=str(last_vap.get("backend", "")),
                turngpt_raw_probability=last_tg.get("raw_probability"),
                turngpt_shaped_probability=last_tg.get("shaped_probability"),
                turngpt_prompt_tokens=last_tg.get("prompt_tokens"),
                turngpt_timeout_ms=getattr(decision, "tg_timeout_ms", None),
            )

        time.sleep(0.02)

    return None


def speak_realtime(
    text: str,
    acoustic_stop_ms: Optional[int],
) -> dict:
    clean = clean_for_speech(text)
    if not clean:
        raise RuntimeError("Empty response after speech cleaning.")

    ASR.pause(True)
    ASR.reset()

    try:
        try:
            FR.set_led("red")
            FR.gaze_hold()
        except Exception:
            pass

        FR.say_text_async(clean)

        if not FR.speaking_started_event().wait(timeout=15.0):
            raise TimeoutError("No Furhat speak-start event within 15 seconds.")

        speech = FR.last_speech_metrics()
        robot_start_ms = speech.get("start_ms")

        if robot_start_ms is None:
            raise RuntimeError("Furhat speak-start event had no start timestamp.")

        fto_ms = (
            robot_start_ms - acoustic_stop_ms
            if acoustic_stop_ms is not None
            else None
        )

        FR.speaking_done_event().wait(timeout=60.0)
        final_metrics = FR.last_speech_metrics()

        return {
            "clean_text": clean,
            "robot_start_ms": robot_start_ms,
            "fto_ms": fto_ms,
            "furhat_gen_time_ms": speech.get("gen_time_ms"),
            "robot_turn_len_ms": final_metrics.get("duration_ms"),
        }

    finally:
        try:
            FR.set_led("off")
            FR.gaze_yield()
        except Exception:
            pass

        ASR.reset()
        ASR.pause(False)


def initial_state() -> InterviewState:
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


def check_stack(condition: str) -> None:
    print("MODEL:", MODEL)
    print("Warming qwen2.5:7b...")
    warmup()
    print("Ollama backend: OK")

    print("Connecting Furhat Realtime...")
    FR.connect()
    print("Furhat Realtime: OK")

    print("Starting ASR...")
    ASR.start_asr_stream()
    ASR.pause(False)
    print("ASR: OK")

    if condition == "vap_turngpt":
        print("Warming VAP...")
        if not VAP.warm_up(seconds=float(getattr(config, "VAP_WARMUP_S", 4.0))):
            raise RuntimeError(f"VAP warm-up failed: {VAP.backend_info()}")
        print("VAP:", VAP.backend_info())

        print("Warming TurnGPT...")
        if not TG.warm_up_turngpt():
            raise RuntimeError(f"TurnGPT warm-up failed: {TG.backend_info()}")
        print("TurnGPT:", TG.backend_info())


def run(condition: str, participant_id: str, max_turns: Optional[int] = None) -> None:
    global _start_time

    if condition not in {"simple", "vap_turngpt"}:
        raise ValueError("condition must be 'simple' or 'vap_turngpt'")

    logger = RQ3BLogger(condition, participant_id)
    state = initial_state()
    turn_index = 0

    try:
        check_stack(condition)
        _start_time = time.time()

        # Same greeting in both conditions.
        greeting_metrics = speak_realtime(GREETING, acoustic_stop_ms=None)
        state["history"].append({"role": "assistant", "content": GREETING})

        while elapsed_s() < TIME_LIMIT_S:
            endpoint = listen_until_endpoint(condition, state)
            if endpoint is None:
                break

            turn_index += 1

            state["last_user_message"] = endpoint.user_text
            state["history"].append(
                {"role": "user", "content": endpoint.user_text}
            )

            safety_before = len(state.get("safety_log", []))

            graph_start = now_ms()
            state = interview_graph.invoke(state)
            graph_end = now_ms()
            graph_latency_ms = graph_end - graph_start

            response_text = state.get("response", "")
            speech = speak_realtime(
                response_text,
                acoustic_stop_ms=endpoint.acoustic_stop_ms,
            )

            state["history"].append(
                {"role": "assistant", "content": response_text}
            )

            safety_after = len(state.get("safety_log", []))
            new_safety = state.get("safety_log", [])[safety_before:safety_after]
            safety_violation_count = sum(
                1 for item in new_safety if item.get("violations")
            )

            logger.write(
                {
                    "participant_id": participant_id,
                    "condition": condition,
                    "turn_index": turn_index,
                    "user_text": endpoint.user_text,
                    "bot_text": response_text,
                    "agent": state.get("next_agent", ""),
                    "topic": state.get("current_topic", ""),
                    "yield_reason": endpoint.reason,
                    "user_start_ms": endpoint.user_start_ms,
                    "acoustic_stop_ms": endpoint.acoustic_stop_ms,
                    "asr_final_ms": endpoint.asr_final_ms,
                    "asr_final_latency_ms":
                        endpoint.asr_final_ms - endpoint.acoustic_stop_ms,
                    "endpoint_permission_ms": endpoint.endpoint_ms,
                    "endpoint_delay_ms": endpoint.endpoint_delay_ms,
                    "graph_latency_ms": graph_latency_ms,
                    "robot_start_ms": speech["robot_start_ms"],
                    "fto_ms": speech["fto_ms"],
                    "furhat_gen_time_ms": speech["furhat_gen_time_ms"],
                    "vap_p_now_user": endpoint.vap_p_now_user,
                    "vap_p_fut_user": endpoint.vap_p_fut_user,
                    "vap_p_now_robot": endpoint.vap_p_now_robot,
                    "vap_p_fut_robot": endpoint.vap_p_fut_robot,
                    "vap_forward_ms": endpoint.vap_forward_ms,
                    "vap_backend": endpoint.vap_backend,
                    "turngpt_raw_probability":
                        endpoint.turngpt_raw_probability,
                    "turngpt_shaped_probability":
                        endpoint.turngpt_shaped_probability,
                    "turngpt_prompt_tokens":
                        endpoint.turngpt_prompt_tokens,
                    "turngpt_timeout_ms":
                        endpoint.turngpt_timeout_ms,
                    "safety_violation_count":
                        safety_violation_count,
                }
            )

            print(
                f"turn={turn_index} condition={condition} "
                f"endpoint={endpoint.endpoint_delay_ms}ms "
                f"graph={graph_latency_ms}ms "
                f"FTO={speech['fto_ms']}ms "
                f"reason={endpoint.reason}"
            )

            if state.get("interview_phase") == "closing":
                break

            if max_turns is not None and turn_index >= max_turns:
                break

        print("RQ3-B session log:", logger.path)

    finally:
        logger.close()

        try:
            ASR.stop_asr_stream()
        except Exception:
            pass

        try:
            FR.disconnect()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        required=True,
        choices=["simple", "vap_turngpt"],
    )
    parser.add_argument(
        "--participant",
        default="DEV",
        help="Development/test ID now; matched experimental ID later.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Use 1 for a short smoke test; omit for a normal session.",
    )
    args = parser.parse_args()

    run(args.condition, args.participant, args.max_turns)


if __name__ == "__main__":
    main()
