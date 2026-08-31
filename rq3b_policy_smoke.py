from rq3b_policy import (
    SIMPLE_SILENCE_MS,
    simple_endpoint_decision,
    predictive_endpoint_decision,
    map_turngpt_timeout,
)

import config


print("RQ3-B POLICY SMOKE TEST")
print("=" * 60)

print("\nCONFIG SNAPSHOT")
print("SIMPLE_SILENCE_MS =", SIMPLE_SILENCE_MS)
print("MIN_GAP_TIME =", config.MIN_GAP_TIME)
print(
    "VAP robot thresholds =",
    config.VAP_PNOW_YIELD_THRESHOLD,
    config.VAP_PFUT_YIELD_THRESHOLD,
)
print(
    "VAP user-low thresholds =",
    config.USER_LOW_NOW_FOR_YIELD,
    config.USER_LOW_FUT_FOR_YIELD,
)
print(
    "TurnGPT ladder =",
    config.TURNGPT_YIELD_TIMEOUTS,
)

# ---------------------------------------------------------
# SIMPLE
# ---------------------------------------------------------

d = simple_endpoint_decision(
    final_ready=True,
    user_speaking=False,
    gap_ms=2499,
)
assert not d.should_yield

d = simple_endpoint_decision(
    final_ready=True,
    user_speaking=False,
    gap_ms=2500,
)
assert d.should_yield
assert d.reason == "fixed_2500ms_silence"

d = simple_endpoint_decision(
    final_ready=False,
    user_speaking=False,
    gap_ms=4000,
)
assert not d.should_yield

d = simple_endpoint_decision(
    final_ready=True,
    user_speaking=True,
    gap_ms=4000,
)
assert not d.should_yield

print("\nSIMPLE tests: PASS")


# ---------------------------------------------------------
# TURNGPT LADDER
# ---------------------------------------------------------

assert map_turngpt_timeout(0.60) == 200
assert map_turngpt_timeout(0.50) == 200
assert map_turngpt_timeout(0.30) == 400
assert map_turngpt_timeout(0.20) == 800
assert map_turngpt_timeout(0.10) == 1600
assert map_turngpt_timeout(0.00) == 2500

print("TurnGPT ladder tests: PASS")


# ---------------------------------------------------------
# VAP GATE
# ---------------------------------------------------------

common = dict(
    final_ready=True,
    user_speaking=False,
    p_now_user=0.20,
    p_future_user=0.20,
    p_now_robot=0.80,
    p_future_robot=0.80,
    tg_score=0.00,
)

d = predictive_endpoint_decision(
    gap_ms=499,
    **common,
)
assert not d.should_yield

d = predictive_endpoint_decision(
    gap_ms=500,
    **common,
)
assert d.should_yield
assert d.reason == "vap_gate"

print("VAP minimum-gap tests: PASS")


# ---------------------------------------------------------
# TURNGPT FALLBACK
# VAP intentionally made non-yielding.
# ---------------------------------------------------------

common_tg = dict(
    final_ready=True,
    user_speaking=False,
    p_now_user=0.90,
    p_future_user=0.90,
    p_now_robot=0.10,
    p_future_robot=0.10,
    tg_score=0.50,
)

d = predictive_endpoint_decision(
    gap_ms=199,
    **common_tg,
)
assert not d.should_yield

d = predictive_endpoint_decision(
    gap_ms=200,
    **common_tg,
)
assert d.should_yield
assert d.reason == "turngpt_timeout"

print("TurnGPT timeout tests: PASS")


# ---------------------------------------------------------
# USER SPEAKING MUST ALWAYS BLOCK YIELD
# ---------------------------------------------------------

d = predictive_endpoint_decision(
    final_ready=True,
    user_speaking=True,
    gap_ms=5000,
    p_now_user=0.05,
    p_future_user=0.05,
    p_now_robot=0.95,
    p_future_robot=0.95,
    tg_score=1.0,
)

assert not d.should_yield
assert d.reason == "user_speaking"

print("Speaking-block test: PASS")


# ---------------------------------------------------------
# FINAL TRANSCRIPT REQUIRED IN BOTH CONDITIONS
# ---------------------------------------------------------

d = predictive_endpoint_decision(
    final_ready=False,
    user_speaking=False,
    gap_ms=5000,
    p_now_user=0.05,
    p_future_user=0.05,
    p_now_robot=0.95,
    p_future_robot=0.95,
    tg_score=1.0,
)

assert not d.should_yield
assert d.reason == "awaiting_final_transcript"

print("Final-transcript gate: PASS")

print("\nRQ3-B POLICY SMOKE: PASS")
