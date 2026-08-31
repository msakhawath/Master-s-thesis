# =============================================================
# multi_agent_main.py  â€”  The New Main Controller
# =============================================================
# This replaces baseline_main.py.
# It keeps the SAME structure (ASR â†’ loop â†’ Furhat) but
# replaces the single LLM call with the multi-agent graph.
#
# HOW TO RUN:
#   python multi_agent_main.py
#
# Make sure:
#   1. Virtual Furhat is running (localhost:54321)
#   2. Ollama is running with the runtime model pulled
#      (qwen2.5:7b â€” defined by MODEL in multi_agent/agents.py)
#   3. Your AssemblyAI API key is set in config.py
# =============================================================

from __future__ import annotations
import time
import logging
from typing import Optional

import config
import asr_module  as ASR
import furhat_module as FR
import tts_module  as TTS
import logger as SESSION_LOG

from multi_agent.graph import interview_graph
from multi_agent.state import InterviewState, INTERVIEW_TOPICS

# â”€â”€ Ollama health check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _check_ollama():
    """
    Verifies Ollama is reachable and the model is loaded
    BEFORE starting the session. Prints a clear error if not.
    """
    from openai import OpenAI as _OAI
    try:
        from multi_agent.agents import MODEL, warmup   # persistent-client warm-up
        c = _OAI(base_url="http://127.0.0.1:11435/v1", api_key="ollama")
        models = [m.id for m in c.models.list().data]
        if not models:
            print("\nâŒ  Ollama is running but NO models are loaded.")
            print(f"    Run:  ollama pull {MODEL}\n")
            return False
        if MODEL not in models:
            print(f"\nâŒ  Model '{MODEL}' not found in Ollama.")
            print(f"    Available: {models}")
            print(f"    Run:  ollama pull {MODEL}\n")
            return False
        # Quick smoke test â€” actually load the model to check RAM
        print(f"  Checking model loads into RAM...")
        try:
            warmup()
        except Exception as e:
            print(f"\nâŒ  Model '{MODEL}' failed to load: {e}")
            print(f"    Your system may not have enough free RAM.")
            print(f"    Try:  ollama pull qwen2.5:7b  (needs only ~1.2GB)\n")
            return False
        print(f"âœ“  Ollama OK â€” model '{MODEL}' is ready.")
        return True
    except Exception as e:
        print(f"\nâŒ  Cannot reach Ollama at localhost:11434 â†’ {e}")
        print("    Make sure Ollama is running:")
        print("    Windows: open Ollama app, or run 'ollama serve' in a terminal\n")
        return False

# â”€â”€ Console logger â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log = logging.getLogger("multi_agent_controller")
log.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(_h)

# â”€â”€ Session settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TIME_LIMIT_S    = getattr(config, "SESSION_TIME_LIMIT_S", 600.0)  # 10 min
GREETING        = "Hi, I am Furhat, your interviewer at DataTech Berlin. We are interviewing for a Junior Data Scientist position. Are you ready to begin?"
SEP             = "|||"   # must match multi_agent/agents.py

# â”€â”€ Timing helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_start_time: Optional[float] = None

def _elapsed() -> float:
    return 0.0 if _start_time is None else time.time() - _start_time

def _now_ms() -> int:
    return int(time.time() * 1000)


# =============================================================
# Speaking: send text to Furhat via TTS
# =============================================================
def _furhat_cmd(fn, *args, **kwargs):
    """
    Calls a Furhat function and silently ignores connection errors.
    This prevents a single LED/gaze failure from blocking the system
    for 12 seconds (3 retries Ã— 4s each) when Furhat is not open.
    """
    try:
        fn(*args, **kwargs)
    except Exception:
        pass   # Furhat not reachable â€” skip silently, don't block


def _clean_for_speech(text: str) -> str:
    """
    Removes ALL separator characters and formatting artefacts
    before sending text to Furhat's TTS.
    Fixes: '|||' spoken aloud, '||' spoken aloud, leading/trailing quotes.
    """
    import re
    # Remove any sequence of pipe characters (|, ||, |||)
    cleaned = re.sub(r'\|+', ' ', text)
    # Remove surrounding quotes the model sometimes adds
    cleaned = cleaned.replace('"', '').replace('"', '').replace('"', '').replace("'", '')
    # Collapse multiple spaces
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    return cleaned.strip()


def _speak(text: str) -> None:
    """
    Converts text â†’ audio â†’ plays on virtual Furhat.
    Pauses ASR while robot is speaking (so it doesn't
    accidentally transcribe its own voice).
    """
    if not text or not text.strip():
        return

    # Strip separators and quotes before sending to TTS
    clean = _clean_for_speech(text)

    log.info(f"[ROBOT] {clean}")

    try:
        ASR.pause(True)
        ASR.reset()
        _furhat_cmd(FR.set_led, "red")    # fail fast, no 12-second retry block
        _furhat_cmd(FR.gaze_hold)

        # Start TTS generation
        TTS.tts_start(clean, make_wav=False, blocking=False)

        # Wait until first audio chunk is ready, then send to Furhat
        try:
            TTS.vap_seen_event().wait()
        except Exception:
            pass

        _furhat_cmd(FR.say_text_async, clean)

        # Wait for Furhat to finish speaking
        try:
            FR.speaking_done_event().wait()
        except Exception:
            pass

    except Exception as e:
        log.error(f"Speaking failed: {e}")
    finally:
        _furhat_cmd(FR.set_led, "off")
        _furhat_cmd(FR.gaze_yield)
        try:
            ASR.reset()
            ASR.pause(False)
        except Exception:
            pass


# =============================================================
# Listening: wait for user to finish speaking
# =============================================================
def _listen() -> Optional[str]:
    """
    Waits until:
      (A) ASR returns a final transcript, AND
      (B) 1.0 second of silence confirmed
    Returns the transcript text, or None on timeout.
    """
    final_text: Optional[str] = None
    silence_start_ms: Optional[int] = None
    user_end_stamped = False
    last_partial_ms: Optional[int] = None

    SILENCE_REQUIRED_MS = 2500   # 2.5s silence â€” gives user time to finish sentence
                                 # (1s was too short, cutting off mid-utterance)

    while True:
        # Check session time limit
        if _elapsed() >= TIME_LIMIT_S:
            return None

        now = _now_ms()

        # Pull ASR results
        if ASR.has_new_result():
            kind, text = ASR.get_result() or (None, None)

            if kind == "partial" and text:
                last_partial_ms = now
                silence_start_ms = None   # reset silence on new speech

            if kind == "final" and text:
                # Accumulate segments (AssemblyAI emits one final per pause) so a
                # multi-sentence answer is captured as one turn, not just the last.
                final_text = (final_text + " " + text).strip() if final_text else text
                silence_start_ms = None   # new speech â†’ restart silence window
                log.info(f"[USER ] {text}")

        # Detect silence
        is_speaking = _detect_speaking(last_partial_ms, now)

        if is_speaking is False:
            if silence_start_ms is None:
                silence_start_ms = now
            if not user_end_stamped and (now - silence_start_ms) >= SILENCE_REQUIRED_MS:
                user_end_stamped = True

        elif is_speaking is True:
            silence_start_ms = None

        # Exit when BOTH conditions met
        if final_text and user_end_stamped:
            return final_text

        time.sleep(0.02)


def _detect_speaking(last_partial_ms, now_ms_val) -> Optional[bool]:
    """Simple speaking detection based on recent partial transcripts."""
    try:
        speaking = ASR.user_speaking()
        if isinstance(speaking, bool):
            return speaking
    except Exception:
        pass
    if last_partial_ms is not None:
        return (now_ms_val - last_partial_ms) <= 200
    return None


# =============================================================
# Main loop
# =============================================================
def run() -> None:
    global _start_time

    # â”€â”€ Initialise all modules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    session_logger = SESSION_LOG.SessionLogger(run_mode="multi_agent")
    session_logger.log_event(event_type="session_start")

    ASR.start_asr_stream()
    ASR.pause(False)
    try:
        FR.set_led("off")
    except Exception:
        pass

    _start_time = time.time()
    log.info("Multi-agent interview system started.")

    # â”€â”€ Initial state for the LangGraph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # This is the starting "notepad" that will be updated
    # every turn as the conversation progresses.
    state: InterviewState = {
        "history":               [],
        "last_user_message":     "",
        "current_topic":         INTERVIEW_TOPICS[0],   # start with education
        "topics_covered":        [],
        "probed_current_topic":  False,
        "interview_phase":       "opening",
        "next_agent":            "interviewer",
        "response":              "",
        "safety_log":            [],   # collects violation data for RQ3
    }

    # â”€â”€ Greeting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _speak(GREETING)
    state["history"].append({"role": "assistant", "content": GREETING})

    # â”€â”€ Main conversation loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    while _elapsed() < TIME_LIMIT_S:

        # 1. Listen to the user
        user_text = _listen()
        if user_text is None:
            # Time limit reached
            break

        # 2. Update the state with what the user just said
        state["last_user_message"] = user_text
        state["history"].append({"role": "user", "content": user_text})

        robot_start_ms = _now_ms()

        # 3. Run the multi-agent graph
        #    The graph runs: Orchestrator â†’ (Interviewer | Prober | Feedback)
        #    and returns the updated state with state["response"] filled in.
        try:
            state = interview_graph.invoke(state)
        except Exception as e:
            log.exception(f"Graph error: {e}")
            state["response"] = "Could you elaborate on that? |||"

        robot_latency_ms = _now_ms() - robot_start_ms
        log.info(f"Agent that ran: {state.get('next_agent','?')}  |  "
                 f"Graph latency: {robot_latency_ms}ms")

        # 4. Speak the response
        response_text = state.get("response", "")
        _speak(response_text)

        # 5. Add robot response to history
        state["history"].append({"role": "assistant", "content": response_text})

        # 6. Log the turn for your thesis data
        session_logger.log_event(
            event_type="agent_turn",
            agent=state.get("next_agent", "unknown"),
            topic=state.get("current_topic", ""),
            user_text=user_text,
            bot_text=response_text,
            graph_latency_ms=robot_latency_ms,
        )

        # 7. If the Feedback agent ran, the interview is done
        if state.get("interview_phase") == "closing":
            log.info("Interview complete. Shutting down.")
            break

    # â”€â”€ Export safety log for RQ3 analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # This CSV gives you the raw data for your failure mode
    # analysis. Every violation the Safety agent caught is here.
    import csv, os
    safety_log = state.get("safety_log", [])
    if safety_log:
        log_path = os.path.join(
            os.path.dirname(__file__),
            f"safety_log_{int(time.time())}.csv"
        )
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=safety_log[0].keys())
            writer.writeheader()
            writer.writerows(safety_log)

        violations_found = [r for r in safety_log if r["violations"]]
        log.info(f"Safety log saved â†’ {log_path}")
        log.info(f"Total turns: {len(safety_log)} | "
                 f"Violations caught: {len(violations_found)} | "
                 f"Clean passes: {len(safety_log) - len(violations_found)}")

        # Print violation summary to console
        if violations_found:
            from collections import Counter
            all_v = [v for r in violations_found for v in r["violations"]]
            counts = Counter(all_v)
            log.info("Violation breakdown: " +
                     ", ".join(f"{k}={v}" for k, v in counts.items()))

    # â”€â”€ Shutdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        FR.stop_text(barge_in=True)
    except Exception:
        pass
    try:
        ASR.stop_asr_stream()
    except Exception:
        pass
    try:
        session_logger.close()
    except Exception:
        pass

    log.info("Session ended.")


# =============================================================
# Entry point
# =============================================================
if __name__ == "__main__":
    if not _check_ollama():
        print("Fix the Ollama issue above, then re-run.")
        exit(1)
    try:
        run()
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C).")
    except Exception as e:
        log.exception(f"FATAL: {e}")

