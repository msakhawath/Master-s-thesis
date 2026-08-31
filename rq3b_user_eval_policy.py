from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import config


class RQ3BCondition(str, Enum):
    SIMPLE = "simple"
    VAP_TURNGPT = "vap_turngpt"


@dataclass(frozen=True)
class EndpointDecision:
    should_yield: bool
    condition: str
    reason: str
    gap_ms: int

    tg_score: Optional[float] = None
    tg_timeout_ms: Optional[int] = None

    p_now_user: Optional[float] = None
    p_future_user: Optional[float] = None
    p_now_robot: Optional[float] = None
    p_future_robot: Optional[float] = None

    vap_gate_ok: Optional[bool] = None


# ------------------------------------------------------------------
# SIMPLE ENDPOINTING
# ------------------------------------------------------------------

SIMPLE_SILENCE_MS = 2500


def simple_endpoint_decision(
    *,
    final_ready: bool,
    user_speaking: bool,
    gap_ms: int,
) -> EndpointDecision:

    gap_ms = max(0, int(gap_ms))

    if not final_ready:
        return EndpointDecision(
            should_yield=False,
            condition=RQ3BCondition.SIMPLE.value,
            reason="awaiting_final_transcript",
            gap_ms=gap_ms,
        )

    if user_speaking:
        return EndpointDecision(
            should_yield=False,
            condition=RQ3BCondition.SIMPLE.value,
            reason="user_speaking",
            gap_ms=gap_ms,
        )

    if gap_ms < SIMPLE_SILENCE_MS:
        return EndpointDecision(
            should_yield=False,
            condition=RQ3BCondition.SIMPLE.value,
            reason="silence_threshold_not_reached",
            gap_ms=gap_ms,
        )

    return EndpointDecision(
        should_yield=True,
        condition=RQ3BCondition.SIMPLE.value,
        reason="fixed_2500ms_silence",
        gap_ms=gap_ms,
    )


# ------------------------------------------------------------------
# TURNGPT TIMEOUT
# ------------------------------------------------------------------

def map_turngpt_timeout(tg_score: float) -> int:
    """
    Map the controller-shaped TurnGPT score to the configured timeout.

    Uses config.TURNGPT_YIELD_TIMEOUTS exactly.
    """

    p = float(tg_score)

    ladder = list(
        getattr(
            config,
            "TURNGPT_YIELD_TIMEOUTS",
            [
                (0.50, 200),
                (0.30, 400),
                (0.20, 800),
                (0.10, 1600),
                (0.00, 2500),
            ],
        )
    )

    for threshold, timeout_ms in sorted(
        ladder,
        key=lambda x: -float(x[0]),
    ):
        if p >= float(threshold):
            return int(timeout_ms)

    return int(ladder[-1][1])


# ------------------------------------------------------------------
# VAP + TURNGPT ENDPOINTING
# ------------------------------------------------------------------

def predictive_endpoint_decision(
    *,
    final_ready: bool,
    user_speaking: bool,
    gap_ms: int,
    p_now_user: float,
    p_future_user: float,
    p_now_robot: float,
    p_future_robot: float,
    tg_score: float,
) -> EndpointDecision:

    gap_ms = max(0, int(gap_ms))

    tg_score = float(tg_score)

    pn_u = float(p_now_user)
    pf_u = float(p_future_user)
    pn_r = float(p_now_robot)
    pf_r = float(p_future_robot)

    tg_timeout_ms = map_turngpt_timeout(tg_score)

    if not final_ready:
        return EndpointDecision(
            should_yield=False,
            condition=RQ3BCondition.VAP_TURNGPT.value,
            reason="awaiting_final_transcript",
            gap_ms=gap_ms,
            tg_score=tg_score,
            tg_timeout_ms=tg_timeout_ms,
            p_now_user=pn_u,
            p_future_user=pf_u,
            p_now_robot=pn_r,
            p_future_robot=pf_r,
            vap_gate_ok=False,
        )

    if user_speaking:
        return EndpointDecision(
            should_yield=False,
            condition=RQ3BCondition.VAP_TURNGPT.value,
            reason="user_speaking",
            gap_ms=gap_ms,
            tg_score=tg_score,
            tg_timeout_ms=tg_timeout_ms,
            p_now_user=pn_u,
            p_future_user=pf_u,
            p_now_robot=pn_r,
            p_future_robot=pf_r,
            vap_gate_ok=False,
        )

    user_unlikely = (
        pn_u <= float(
            getattr(
                config,
                "USER_LOW_NOW_FOR_YIELD",
                0.47,
            )
        )
        and
        pf_u <= float(
            getattr(
                config,
                "USER_LOW_FUT_FOR_YIELD",
                0.46,
            )
        )
    )

    robot_likely = (
        pn_r >= float(
            getattr(
                config,
                "VAP_PNOW_YIELD_THRESHOLD",
                0.48,
            )
        )
        and
        pf_r >= float(
            getattr(
                config,
                "VAP_PFUT_YIELD_THRESHOLD",
                0.48,
            )
        )
    )

    min_gap_ms = int(
        getattr(
            config,
            "MIN_GAP_TIME",
            500,
        )
    )

    # Pilot safety correction for natural interview answers:
    # the TurnGPT timeout must NOT bypass a minimum silence hold.
    # 1000 ms is intentionally conservative for the formative user pilot.
    predictive_min_gap_ms = max(
        min_gap_ms,
        int(getattr(config, "RQ3B_USER_EVAL_MIN_GAP_MS", 1000)),
    )

    vap_gate_ok = (
        (user_unlikely or robot_likely)
        and
        gap_ms >= predictive_min_gap_ms
    )

    tg_timeout_ok = (
        gap_ms >= max(tg_timeout_ms, predictive_min_gap_ms)
    )

    if vap_gate_ok:
        reason = "vap_gate"
        should_yield = True

    elif tg_timeout_ok:
        reason = "turngpt_timeout_guarded"
        should_yield = True

    else:
        reason = "predictive_wait"
        should_yield = False

    return EndpointDecision(
        should_yield=should_yield,
        condition=RQ3BCondition.VAP_TURNGPT.value,
        reason=reason,
        gap_ms=gap_ms,
        tg_score=tg_score,
        tg_timeout_ms=tg_timeout_ms,
        p_now_user=pn_u,
        p_future_user=pf_u,
        p_now_robot=pn_r,
        p_future_robot=pf_r,
        vap_gate_ok=vap_gate_ok,
    )


def decide_endpoint(
    condition: str,
    **kwargs,
) -> EndpointDecision:

    condition = str(condition).strip().lower()

    if condition == RQ3BCondition.SIMPLE.value:
        return simple_endpoint_decision(**kwargs)

    if condition == RQ3BCondition.VAP_TURNGPT.value:
        return predictive_endpoint_decision(**kwargs)

    raise ValueError(
        f"Unknown RQ3-B condition: {condition!r}"
    )
