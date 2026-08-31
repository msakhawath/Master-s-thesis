# ============================================================
# logger.py — thesis experiment CSV logger
# ============================================================

from __future__ import annotations

import csv
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    import config
    LOGS_DIR = Path(config.ROOT_DIR) / "logs"
except Exception:
    LOGS_DIR = Path(__file__).resolve().parent / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


class SessionLogger:
    """
    Unified event logger for baseline, multi-agent, and turn-taking experiments.

    Important:
    - Absolute speech-anchor timestamps are stored separately from derived latency.
    - Multi-agent metadata is retained for RQ3.
    - VAP/TurnGPT diagnostics are available for later component-ablation analysis.
    - Unknown legacy keyword arguments are tolerated for backward compatibility.
    """

    FIELDNAMES = [
        "timestamp_ms",
        "participant_id",
        "session_id",
        "run_mode",
        "turn_id",
        "event_type",

        "user_text",
        "bot_text",

        # Derived timing
        "response_time_ms",
        "fto_ms",
        "user_turn_len_ms",
        "robot_turn_len_ms",

        # Interruptions / overlap
        "interrupt_flag",
        "barge_in_stop",
        "false_cutin_flag",
        "barge_in_success_flag",
        "overlap_prev_turn",
        "backchannel_overlap_flag",
        "fallback_flag",

        # Latency breakdown
        "asr_final_latency_ms",
        "llm_latency_ms",
        "tts_latency_ms",
        "graph_latency_ms",
        "furhat_gen_time_ms",

        # Absolute speech anchors
        "user_stop_speech",
        "robot_start_speech",

        # Multi-agent / RQ3
        "agent",
        "topic",
        "safety_violations",

        # VAP diagnostics
        "vap_p_now_user",
        "vap_p_fut_user",
        "vap_p_now_robot",
        "vap_p_fut_robot",
        "vap_forward_ms",
        "vap_backend",

        # TurnGPT diagnostics
        "turngpt_raw_probability",
        "turngpt_shaped_probability",
        "turngpt_prompt_tokens",
    ]

    def __init__(
        self,
        run_mode: str,
        participant_id: Optional[str] = None,
    ):
        self.run_mode = (run_mode or "").lower().strip() or "turntaking"

        self.participant_id = (
            participant_id.strip()
            if participant_id
            else f"P{str(uuid.uuid4())[:8]}"
        )

        self.session_id = (
            f"{self.participant_id}_{self.run_mode}_{int(time.time())}"
        )

        self.file_path = LOGS_DIR / f"{self.session_id}.csv"

        self.file = self.file_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDNAMES,
            extrasaction="ignore",
        )
        self.writer.writeheader()

        self.turn_counter = 0

    @staticmethod
    def _ts() -> int:
        return int(time.time() * 1000)

    def log_event(
        self,
        event_type: str,
        user_text: Optional[str] = "",
        bot_text: Optional[str] = "",
        response_time_ms: Optional[float] = None,
        fto_ms: Optional[float] = None,
        user_turn_len_ms: Optional[float] = None,
        robot_turn_len_ms: Optional[float] = None,
        interrupt_flag: bool = False,
        overlap_prev_turn: Optional[int] = 0,
        barge_in_stop: Optional[int] = 0,
        false_cutin_flag: Optional[int] = 0,
        barge_in_success_flag: Optional[int] = 0,
        backchannel_overlap_flag: Optional[int] = 0,
        fallback_flag: Optional[int] = 0,
        asr_final_latency_ms: Optional[float] = None,
        llm_latency_ms: Optional[float] = None,
        tts_latency_ms: Optional[float] = None,
        graph_latency_ms: Optional[float] = None,
        furhat_gen_time_ms: Optional[float] = None,
        user_stop_speech: Optional[int] = None,
        robot_start_speech: Optional[int] = None,
        agent: Optional[str] = "",
        topic: Optional[str] = "",
        safety_violations: Optional[str] = "",
        vap_p_now_user: Optional[float] = None,
        vap_p_fut_user: Optional[float] = None,
        vap_p_now_robot: Optional[float] = None,
        vap_p_fut_robot: Optional[float] = None,
        vap_forward_ms: Optional[float] = None,
        vap_backend: Optional[str] = "",
        turngpt_raw_probability: Optional[float] = None,
        turngpt_shaped_probability: Optional[float] = None,
        turngpt_prompt_tokens: Optional[int] = None,
        **kwargs,
    ):
        # Backward-compatible anchor names
        if "user_stop_ms" in kwargs and user_stop_speech is None:
            user_stop_speech = kwargs.pop("user_stop_ms")

        if "robot_start_ms" in kwargs and robot_start_speech is None:
            robot_start_speech = kwargs.pop("robot_start_ms")

        # Keep legacy counter behavior
        if event_type in ("user_final", "robot_start"):
            self.turn_counter += 1

        def blank_if_none(value):
            return "" if value is None else value

        row = {
            "timestamp_ms": self._ts(),
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "run_mode": self.run_mode,
            "turn_id": self.turn_counter,
            "event_type": event_type,

            "user_text": (user_text or "").strip(),
            "bot_text": (bot_text or "").strip(),

            "response_time_ms": blank_if_none(response_time_ms),
            "fto_ms": blank_if_none(fto_ms),
            "user_turn_len_ms": blank_if_none(user_turn_len_ms),
            "robot_turn_len_ms": blank_if_none(robot_turn_len_ms),

            "interrupt_flag": bool(interrupt_flag),
            "barge_in_stop": int(barge_in_stop or 0),
            "false_cutin_flag": int(false_cutin_flag or 0),
            "barge_in_success_flag": int(barge_in_success_flag or 0),
            "overlap_prev_turn": int(overlap_prev_turn or 0),
            "backchannel_overlap_flag": int(backchannel_overlap_flag or 0),
            "fallback_flag": int(fallback_flag or 0),

            "asr_final_latency_ms": blank_if_none(asr_final_latency_ms),
            "llm_latency_ms": blank_if_none(llm_latency_ms),
            "tts_latency_ms": blank_if_none(tts_latency_ms),
            "graph_latency_ms": blank_if_none(graph_latency_ms),
            "furhat_gen_time_ms": blank_if_none(furhat_gen_time_ms),

            "user_stop_speech": blank_if_none(user_stop_speech),
            "robot_start_speech": blank_if_none(robot_start_speech),

            "agent": (agent or "").strip(),
            "topic": (topic or "").strip(),
            "safety_violations": (safety_violations or "").strip(),

            "vap_p_now_user": blank_if_none(vap_p_now_user),
            "vap_p_fut_user": blank_if_none(vap_p_fut_user),
            "vap_p_now_robot": blank_if_none(vap_p_now_robot),
            "vap_p_fut_robot": blank_if_none(vap_p_fut_robot),
            "vap_forward_ms": blank_if_none(vap_forward_ms),
            "vap_backend": (vap_backend or "").strip(),

            "turngpt_raw_probability": blank_if_none(
                turngpt_raw_probability
            ),
            "turngpt_shaped_probability": blank_if_none(
                turngpt_shaped_probability
            ),
            "turngpt_prompt_tokens": blank_if_none(
                turngpt_prompt_tokens
            ),
        }

        self.writer.writerow(row)
        self.file.flush()

    def log_robot_start(
        self,
        bot_text: str,
        fto_ms: Optional[float],
        overlap_prev_turn: int = 0,
        response_time_ms: Optional[float] = None,
        tts_latency_ms: Optional[float] = None,
        user_stop_speech: Optional[int] = None,
        robot_start_speech: Optional[int] = None,
        user_stop_ms: Optional[int] = None,
        robot_start_ms: Optional[int] = None,
        furhat_gen_time_ms: Optional[float] = None,
    ):
        if user_stop_speech is None:
            user_stop_speech = user_stop_ms

        if robot_start_speech is None:
            robot_start_speech = robot_start_ms

        self.log_event(
            event_type="robot_start",
            bot_text=bot_text,
            fto_ms=fto_ms,
            response_time_ms=(
                response_time_ms
                if response_time_ms is not None
                else fto_ms
            ),
            overlap_prev_turn=overlap_prev_turn,
            tts_latency_ms=tts_latency_ms,
            user_stop_speech=user_stop_speech,
            robot_start_speech=robot_start_speech,
            furhat_gen_time_ms=furhat_gen_time_ms,
        )

    def log_robot_end(
        self,
        robot_turn_len_ms: Optional[float],
    ):
        self.log_event(
            event_type="robot_end",
            robot_turn_len_ms=robot_turn_len_ms,
        )

    def log_user_final(
        self,
        user_text: str,
        user_turn_len_ms: Optional[float],
        asr_final_latency_ms: Optional[float] = None,
    ):
        self.log_event(
            event_type="user_final",
            user_text=user_text,
            user_turn_len_ms=user_turn_len_ms,
            asr_final_latency_ms=asr_final_latency_ms,
        )

    def log_barge_in_cut(self):
        self.log_event(
            event_type="barge_in_cut",
            interrupt_flag=True,
            barge_in_stop=1,
        )

    def log_barge_in_success(self):
        self.log_event(
            event_type="barge_in_success",
            barge_in_success_flag=1,
        )

    def log_false_cutin(self):
        self.log_event(
            event_type="false_cutin",
            false_cutin_flag=1,
        )

    def log_backchannel_overlap(self):
        self.log_event(
            event_type="backchannel_overlap",
            backchannel_overlap_flag=1,
        )

    def log_fallback(self, bot_text: str = ""):
        self.log_event(
            event_type="fallback",
            fallback_flag=1,
            bot_text=bot_text,
        )

    def log_llm_ready(
        self,
        llm_latency_ms: Optional[float],
    ):
        self.log_event(
            event_type="llm_ready",
            llm_latency_ms=llm_latency_ms,
        )

    def close(self):
        if not self.file.closed:
            self.file.close()
