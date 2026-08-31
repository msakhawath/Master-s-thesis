r"""
Acceptance / session-quality check for a single recorded session.

    python user_study\check_session.py <session_log.csv> [--max-gap-s 6]

Works with both study arms:
  Condition A -> rq3_results\rq3b_vap_turngpt_<P>_<stamp>.csv   (per-turn schema)
  Condition B -> logs\<session>.csv                              (event schema)

Dead air is measured differently per schema, because the row meanings differ:
  per-turn schema -> one row IS a whole turn, so inter-row gaps include the
                     participant speaking. Responsiveness is fto_ms.
  event  schema   -> rows are individual events, so inter-event gaps are
                     meaningful and a large one indicates a backend stall.

Use it two ways:
  1. ACCEPTANCE, before P01: three consecutive sessions must pass.
  2. QUALITY CONTROL, after every participant, to catch a silent failure before
     the next person is booked.

Exit code 0 = session acceptable.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path


def _f(row: dict, *names):
    for n in names:
        v = (row.get(n) or "").strip()
        if v:
            try:
                return float(v)
            except ValueError:
                pass
    return None


def load(path: Path):
    with path.open(encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def analyse(path: Path, max_gap_s: float):
    rows = load(path)
    if not rows:
        return ["log file is EMPTY - the session produced no rows"], [], {}

    cols = set(rows[0].keys())
    per_turn = "endpoint_delay_ms" in cols          # condition A schema
    problems: list[str] = []   # hard failures - session is not usable
    warnings: list[str] = []   # advisory - investigate, but do not discard
    stats: dict[str, object] = {}

    # ---- robot turns and FTO (shared metric across both arms) ----------------
    if per_turn:
        turns = [r for r in rows if (r.get("bot_text") or "").strip()]
        ftos = [v for v in (_f(r, "fto_ms") for r in turns) if v is not None]
        stamps = [v for v in (_f(r, "timestamp_ms") for r in rows) if v is not None]
        endpoints = [v for v in (_f(r, "endpoint_delay_ms") for r in turns) if v is not None]
        if endpoints:
            stats["endpoint_delay_ms median"] = round(st.median(endpoints), 1)
        reasons: dict[str, int] = {}
        for r in turns:
            k = (r.get("yield_reason") or "?").strip()
            reasons[k] = reasons.get(k, 0) + 1
        stats["yield_reasons"] = reasons
    else:
        turns = [r for r in rows if (r.get("event_type") or "") == "robot_start"]
        ftos = [v for v in (_f(r, "fto_ms") for r in turns) if v is not None]
        stamps = [v for v in (_f(r, "timestamp_ms") for r in rows) if v is not None]

    stats["schema"] = "per-turn (Condition A)" if per_turn else "event (Condition B)"
    stats["rows"] = len(rows)
    stats["robot turns"] = len(turns)
    if ftos:
        s = sorted(ftos)
        stats["fto_ms median"] = round(st.median(ftos), 1)
        stats["fto_ms p95"] = round(s[min(len(s) - 1, int(0.95 * len(s)))], 1)
        stats["fto_ms max"] = round(max(ftos), 1)

    # ---- dead air ------------------------------------------------------------
    # fto_ms is the only sound dead-air measure in BOTH schemas: it is the wait
    # the participant actually experienced between finishing speaking and the
    # robot starting. Raw inter-row / inter-event gaps are NOT usable - they
    # include the participant talking and thinking, which produced false alarms.
    if ftos:
        slow = [v for v in ftos if v / 1000.0 > max_gap_s]
        if slow:
            # WARNING, not a failure. No scientifically justified exclusion
            # threshold for FTO has been established - 6 s is a review heuristic,
            # not a criterion - so slow responses must not silently invalidate a
            # participant session. Investigate before freezing; do not discard.
            warnings.append(
                f"{len(slow)}/{len(ftos)} turn(s) with FTO over {max_gap_s}s "
                f"(worst {max(ftos)/1000.0:.1f}s) - high participant-facing latency; "
                "investigate before study freeze"
            )
    else:
        problems.append("no fto_ms values logged - cannot assess responsiveness")

    if len(stamps) >= 2:  # informational only, never a pass/fail criterion
        stamps.sort()
        stats["largest inter-event gap (s), FYI"] = round(
            max((b - a) / 1000.0 for a, b in zip(stamps, stamps[1:])), 1)

    # ---- latency breakdown and generation efficiency -------------------------
    if not per_turn:
        ev = lambda name: sum(1 for r in rows if (r.get("event_type") or "") == name)

        llm = [v for v in (_f(r, "llm_latency_ms") for r in rows) if v is not None]
        tts = [v for v in (_f(r, "tts_latency_ms") for r in rows) if v is not None]
        if llm:
            stats["llm_latency_ms median"] = round(st.median(llm), 1)
            stats["llm_latency_ms max"] = round(max(llm), 1)
        if tts:
            stats["tts_latency_ms median"] = round(st.median(tts), 1)

        finals, ready = ev("user_final"), ev("llm_ready")
        stats["ASR final segments"] = finals
        stats["LLM generations (llm_ready)"] = ready
        stats["closing_delivered"] = ev("closing_delivered")
        stats["fallback events"] = ev("fallback")

        # THE diagnostic ratio. Before the turn-settled fix this was 30 generations
        # for 21 spoken turns, because every ASR fragment restarted generation.
        # It should now sit at roughly 1.0 - one generation per conversational turn.
        if turns:
            ratio = ready / len(turns)
            stats["generations per robot turn"] = round(ratio, 2)
            if ratio > 1.5:
                problems.append(
                    f"{ready} generations for {len(turns)} robot turns (ratio {ratio:.2f}) - "
                    "generation is still restarting within turns; expect ~1.0"
                )

        bc = ev("backchannel_overlap")
        if bc:
            stats["backchannel_overlap events"] = bc
            if bc > 3 * max(1, len(turns)):
                warnings.append(
                    f"{bc} backchannel-overlap events across {len(turns)} robot turns - "
                    "VAP is probably hearing the robot's own loudspeaker, not the "
                    "participant. Do NOT report these as participant backchannels."
                )

    # ---- completeness --------------------------------------------------------
    if len(turns) < 5:
        problems.append(f"only {len(turns)} robot turns - session likely did not run to completion")

    fallback = sum(1 for r in rows if "did not catch that" in (r.get("bot_text") or "").lower())
    if fallback:
        stats["fallback lines spoken"] = fallback
        problems.append(f"{fallback} backend-failure fallback line(s) spoken - check the console log")

    empty_bot = sum(1 for r in turns if not (r.get("bot_text") or "").strip())
    if empty_bot:
        problems.append(f"{empty_bot} robot turn(s) with empty text")

    return problems, warnings, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--max-gap-s", type=float, default=6.0)
    ap.add_argument("--baseline", help="earlier session log to compare against, "
                                       "e.g. the pre-fix pilot")
    args = ap.parse_args()

    path = Path(args.logfile)
    if not path.exists():
        print(f"FAIL  no such file: {path}")
        return 1

    problems, warnings, stats = analyse(path, args.max_gap_s)

    print(f"\nSESSION CHECK - {path.name}\n" + "=" * 60)
    for k, v in stats.items():
        print(f"  {k:32s} {v}")

    if args.baseline:
        base = Path(args.baseline)
        if not base.exists():
            print(f"\n  (baseline not found: {base})")
        else:
            _, _, bstats = analyse(base, args.max_gap_s)
            print(f"\nCOMPARED WITH {base.name}\n" + "-" * 60)
            print(f"  {'metric':32s} {'baseline':>12s} {'this run':>12s} {'change':>12s}")
            for k in stats:
                a, b = bstats.get(k), stats.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    delta = b - a
                    pct = f"{delta / a * 100:+.0f}%" if a else "n/a"
                    print(f"  {k:32s} {a:>12.1f} {b:>12.1f} {pct:>12s}")

    print("-" * 60)

    for w in warnings:
        print(f"  WARNING: {w}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    if not warnings and not problems:
        print("  no problems detected")

    print("=" * 60)
    if problems:
        print("SESSION NOT ACCEPTABLE - investigate before the next participant.\n")
        return 1
    if warnings:
        print("SESSION ACCEPTABLE, WITH WARNINGS - review them before the study freeze.\n")
        return 0
    print("SESSION ACCEPTABLE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
