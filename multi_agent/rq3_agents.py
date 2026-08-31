# =============================================================
# multi_agent/agents.py  â€”  The Five Specialist Agents
# =============================================================
# Each agent is a Python function that reads the shared state and writes an
# updated state. The specialist generators use qwen2.5:7b through Ollama with
# focused prompts; the Orchestrator is rule-based and makes no LLM call.
#
# Agent pipeline per turn:
#   Orchestrator â†’ Interviewer | Prober | Feedback â†’ Safety/Policy
#
# This is the key difference from the baseline:
#   BASELINE:    one giant prompt handles everything
#   MULTI-AGENT: each agent has a short, focused prompt
#                â†’ faster, more reliable, easier to debug
# =============================================================

from openai import OpenAI
from multi_agent.rq3_state import InterviewState, INTERVIEW_TOPICS

# â”€â”€ Ollama client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ollama runs locally on your computer and exposes an API
# that looks identical to OpenAI's API. So we use the OpenAI
# library but point it at localhost instead of the cloud.
_client = OpenAI(
    base_url="http://127.0.0.1:11435/v1",
    api_key="ollama",
)

MODEL = "qwen2.5:7b"     # 1B variant â€” fits in ~1.2GB RAM for local deployment
                        # gemma3:4b was validated in benchmarking as the optimal
                        # quality model; 1B is used here for hardware feasibility
SEP   = "|||"           # sentence separator (same as benchmark)


import logging as _logging
_llm_log = _logging.getLogger("llm_call")

def _trim_history(history: list, max_turns: int = 6) -> list:
    """
    Keep only the last 6 messages (3 exchanges).
    Prevents prompt from growing every turn â€” that was causing
    latency to increase from 13s â†’ 38s â†’ 79s progressively.
    """
    return history[-max_turns:] if len(history) > max_turns else history


def warmup() -> None:
    """Warm the persistent Ollama client before the interview."""
    _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=1,
        extra_body={"keep_alive": "30m"},
    )


def _call_llm(system_prompt: str, user_message: str, max_tokens: int = 100) -> str:
    """
    A simple helper that calls the configured MODEL (qwen2.5:7b) and returns the full response.
    Every agent uses this â€” they just pass different prompts.
    """
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Log the REAL error so you can debug it
        _llm_log.error(f"LLM call failed â†’ model={MODEL} error={type(e).__name__}: {e}")
        return f"Could you elaborate on that? {SEP}"


# =============================================================
# AGENT 1: Orchestrator
# =============================================================
# The manager. It reads the conversation and decides:
#   → Is the answer good enough? → move to next topic (Interviewer)
#   → Is the answer too vague?   → probe deeper (Prober)
#   → Are all topics done?        → close the interview (Feedback)
#
# IMPORTANT: The Orchestrator does NOT generate any speech.
# It only makes a routing DECISION. The actual words come
# from one of the three specialist agents below.
# =============================================================

# Phrases that signal the user wants to end the interview
_EXIT_PHRASES = [
    # Time / availability
    "not interested", "have to go", "i need to go", "i must go",
    "don't have time", "no time", "not now", "maybe later",
    "i have to leave", "i'm leaving", "don't have any time",
    "don't have time to contact", "no time to contact",
    # Stopping the interview
    "finish the interview", "end the interview", "stop the interview",
    "close this interview", "close the interview", "let's close",
    "finish it", "please finish", "end it", "stop it",
    "wrap up", "wrap it up", "conclude", "let's conclude",
    # Polite refusals
    "can we finish", "can i finish", "want to stop", "need to stop",
    "no thank you", "no thanks", "don't want to", "i'm done",
    "that's all", "goodbye", "bye", "not going to answer",
    "i don't want to continue", "i cannot continue",
    "not going to continue", "i want to leave",
]

def _wants_to_exit(msg: str) -> bool:
    """Returns True if the user is signalling they want to end the interview."""
    return any(phrase in msg for phrase in _EXIT_PHRASES)


def orchestrator_agent(state: InterviewState) -> InterviewState:
    """
    Routes to the correct agent using fast rules â€” NO LLM call.
    This keeps latency low by removing one LLM call per turn.
    """
    last_msg      = (state.get("last_user_message") or "").lower().strip()
    topics_done   = state.get("topics_covered", [])
    phase         = state.get("interview_phase", "opening")
    probed        = state.get("probed_current_topic", False)
    current_topic = state.get("current_topic", "")

    # Rule 0: EXIT INTENT â€” user wants to stop â†’ close immediately
    # Must be checked BEFORE anything else so the robot never
    # ignores a user who says "I have to go" or "no thank you".
    if phase != "opening" and _wants_to_exit(last_msg):
        return {**state, "next_agent": "feedback", "interview_phase": "closing"}

    # Rule 1: Opening phase â†’ always start with Interviewer
    if phase == "opening":
        return {**state, "next_agent": "interviewer", "interview_phase": "main"}

    # Rule 2: All topics covered â†’ close the interview
    remaining = [t for t in INTERVIEW_TOPICS if t not in topics_done]
    if not remaining:
        return {**state, "next_agent": "feedback", "interview_phase": "closing"}

    # Rule 3: Probe if answer seems weak (pure rules, no LLM needed)
    word_count   = len(last_msg.split())
    vague_phrases = [
        "i don't know", "not sure", "not really", "i think", "i guess",
        "no experience", "not familiar", "never done", "maybe", "not much",
    ]
    is_too_short = word_count < 8
    is_vague     = any(p in last_msg for p in vague_phrases)

    if (is_too_short or is_vague) and not probed:
        return {**state, "next_agent": "prober"}

    # Rule 4: Answer is good enough â†’ move to next topic
    new_covered = topics_done + (
        [current_topic] if current_topic and current_topic not in topics_done else []
    )
    # Recompute remaining AFTER adding the current topic. Otherwise the
    # just-finished topic is still first in `remaining`, so next_topic would
    # equal current_topic and the Interviewer would ask the same topic twice.
    next_remaining = [t for t in INTERVIEW_TOPICS if t not in new_covered]
    if not next_remaining:
        # That was the last topic â†’ close the interview.
        return {
            **state,
            "next_agent":      "feedback",
            "interview_phase": "closing",
            "topics_covered":  new_covered,
        }
    return {
        **state,
        "next_agent":           "interviewer",
        "current_topic":        next_remaining[0],
        "topics_covered":       new_covered,
        "probed_current_topic": False,
    }


# =============================================================
# AGENT 2: Interviewer
# =============================================================
# Asks ONE focused question about the current topic.
# Its prompt is SHORT â€” just "ask a question about X".
# This is much faster and more reliable than asking one giant
# LLM to handle the whole interview.
# =============================================================

def interviewer_agent(state: InterviewState) -> InterviewState:
    """
    Asks one interview question about the current topic.
    Writes the question into state["response"].
    """
    topic   = state.get("current_topic", "your experience")
    history = _trim_history(state.get("history", []))

    # Build a short conversation summary so the agent
    # doesn't repeat a question that was already asked.
    recent = ""
    if history:
        recent = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

    if state.get("rq3_condition", "full") == "no_persona":
        interviewer_intro = """You are a job interviewer at DataTech Berlin.
You are interviewing for a Junior Data Scientist role."""
    else:
        interviewer_intro = """You are a warm, professional job interviewer at DataTech Berlin.
You are interviewing for a Junior Data Scientist role."""

    INTERVIEWER_PROMPT = f"""{interviewer_intro}

Task: ask EXACTLY ONE question about the topic: "{topic}".

Strict output rules:
- Produce exactly ONE short sentence containing exactly ONE interview question.
- Keep the whole response under 30 words.
- A brief filler is optional; if used, keep it inside the same sentence, e.g. "Interesting, how did you evaluate the model?"
- Put the exact delimiter {SEP} ONCE, at the very end of the response.
- Do not output any other pipe characters.
- Do not use quotation marks, lists, bullets, headings, or meta-commentary.
- Do not repeat a question already visible in the recent conversation.

Recent conversation:
{recent if recent else "(start of interview)"}"""

    question = _call_llm(INTERVIEWER_PROMPT, f"Ask about: {topic}")
    return {**state, "response": question}


# =============================================================
# AGENT 3: Prober
# =============================================================
# The Orchestrator sends the conversation here when the
# candidate gives a vague or too-short answer.
# The Prober uses the STAR method (Situation, Task, Action,
# Result) to dig deeper before moving to the next topic.
# =============================================================

def prober_agent(state: InterviewState) -> InterviewState:
    """
    Asks a STAR-method follow-up question about the last answer.
    Writes the follow-up into state["response"].
    Sets probed_current_topic = True so we don't probe twice.
    """
    last_msg = state.get("last_user_message", "")
    topic    = state.get("current_topic", "your experience")

    if state.get("rq3_condition", "full") == "no_persona":
        prober_intro = f"""You are a job interviewer. The candidate gave a vague or short answer about "{topic}"."""
    else:
        prober_intro = f"""You are a warm professional job interviewer. The candidate gave a vague or short answer about "{topic}"."""

    PROBER_PROMPT = f"""{prober_intro}

Task: ask EXACTLY ONE focused follow-up question that elicits one useful STAR detail (Situation, Task, Action, or Result).

Strict output rules:
- Produce exactly ONE short sentence containing exactly ONE question.
- Keep the whole response under 25 words.
- Put the exact delimiter {SEP} ONCE, at the very end.
- Do not output any other pipe characters.
- Do not use quotation marks, lists, headings, or meta-commentary."""

    follow_up = _call_llm(PROBER_PROMPT, f"Candidate said: {last_msg}")
    return {**state, "response": follow_up, "probed_current_topic": True}


# =============================================================
# AGENT 4: Feedback
# =============================================================
# Called when all topics are done. Gives the candidate a brief,
# professional closing statement. Does NOT reveal a decision.
# =============================================================

def feedback_agent(state: InterviewState) -> InterviewState:
    """
    Gives a professional closing statement to end the interview.
    """
    topics_done = state.get("topics_covered", [])

    if state.get("rq3_condition", "full") == "no_persona":
        feedback_intro = "You are closing a job interview. Speak directly to the candidate."
    else:
        feedback_intro = "You are closing a professional job interview. Speak directly to the candidate."

    FEEDBACK_PROMPT = f"""{feedback_intro}

Task: give ONE brief closing statement. Ask ZERO questions. Do not reveal a hiring decision.

Strict output rules:
- Produce exactly ONE short sentence under 30 words.
- Thank the candidate and say the responses will be reviewed.
- Put the exact delimiter {SEP} ONCE, at the very end.
- Do not output any other pipe characters.
- Do not use quotation marks, lists, headings, or meta-commentary."""

    closing = _call_llm(FEEDBACK_PROMPT, "Say the closing words now.", max_tokens=100)
    return {**state, "response": closing}


# =============================================================
# AGENT 5: Safety / Policy Controller
# =============================================================
# Final quality gate before the robot speaks.
#
# Deterministic checks repair formatting/length/question-count failures without
# another generation call. A small LLM safety check is used only when a cheap
# response-only topic heuristic finds no interview-related signal.
#
# Every clean pass and every repair is recorded in state["safety_log"].  Keep
# the RAW model response in the log so RQ3 measures the model's actual failure,
# not a response that was silently normalised first.
# =============================================================

# Allowed interview-response signals for the cheap off-topic screen.
_ALLOWED_TOPICS = [
    # Data Science domain keywords
    "data", "machine learning", "python", "model", "algorithm", "neural",
    "deep learning", "sql", "database", "visualization", "pandas", "numpy",
    "sklearn", "tensorflow", "pytorch", "statistics", "regression", "classification",
    "clustering", "feature", "preprocessing", "cleaning", "pipeline", "accuracy",
    "precision", "recall", "f1", "overfitting", "underfitting", "training",
    "dataset", "analysis", "insight", "pattern", "prediction", "jupyter",
    "notebook", "github", "api", "cloud", "spark", "hadoop", "tableau", "power bi",
    # General interview keywords
    "experience", "education", "skill", "work", "team", "communication",
    "problem", "goal", "career", "interview", "company", "datatech", "berlin",
    "challenge", "situation", "result", "action", "project", "background",
    # Conversational / closing / clarification keywords
    "tell me", "describe", "explain", "example", "elaborate", "thank",
    "review", "response", "ready", "begin", "understand", "repeat", "clarify",
    "hear", "listen", "okay", "great", "interesting", "thanks", "sorry",
]

# The generators target <30 words.  A 35-word safety ceiling gives a small
# tolerance while still catching meaningful verbosity; the previous 60-word
# threshold was too far from the actual generation contract.
_SAFETY_MAX_WORDS = 35

_QUESTION_START_RE = __import__("re").compile(
    r"^(?:who|what|when|where|why|how|which|can|could|would|will|do|does|did|"
    r"is|are|was|were|have|has|had|tell me|describe|explain|share|"
    r"walk me through|give me)\b",
    __import__("re").IGNORECASE,
)
_FILLER_PREFIX_RE = __import__("re").compile(
    r"^(?:okay|alright|great|interesting|thanks(?: for sharing)?|sounds good|i see|wonderful)\s*[,!:-]?\s*",
    __import__("re").IGNORECASE,
)
_SECOND_QUESTION_CLAUSE_RE = __import__("re").compile(
    r"\b(?:and|or)\s+(?:who|what|when|where|why|how|which|can|could|would|do|does|did|is|are|was|were)\b",
    __import__("re").IGNORECASE,
)


def _strip_outer_quotes(text: str) -> str:
    import re
    t = (text or "").strip()
    t = re.sub(r'^["â€œâ€˜]+|["â€â€™]+$', '', t).strip()
    return t.replace('"', '').replace('â€œ', '').replace('â€', '')


def _question_units(text: str) -> int:
    """Heuristic count of interview questions/requests.

    Question marks are counted directly.  Imperative interview prompts such as
    'Tell me about...' are counted too, because a 1B model sometimes omits '?'.
    """
    import re
    cleaned = re.sub(r"\|+", "\n", text or "")
    segments = [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+", cleaned) if s.strip()]
    total = 0
    for seg in segments:
        stripped = _FILLER_PREFIX_RE.sub("", seg).strip()
        qmarks = stripped.count("?")
        if qmarks:
            total += qmarks + len(_SECOND_QUESTION_CLAUSE_RE.findall(stripped))
        elif _QUESTION_START_RE.search(stripped):
            total += 1 + len(_SECOND_QUESTION_CLAUSE_RE.findall(stripped))
    return total


def _first_question_unit(text: str) -> str:
    """
    Return only the first question unit according to the SAME
    compound-question heuristic used by _question_units().

    Examples:
        "What did you do? What happened next?"
            -> "What did you do?"

        "How do you evaluate a model, and what metrics do you use?"
            -> "How do you evaluate a model?"

    The Safety controller later appends the canonical ||| delimiter.
    """
    t = _strip_outer_quotes((text or "").strip())

    if not t:
        return ""

    # Separator material is not legitimate question content.
    if "|" in t:
        t = t.split("|", 1)[0].rstrip()

    # Compound second interrogative clause.
    # Uses the same regex as _question_units().
    m = _SECOND_QUESTION_CLAUSE_RE.search(t)

    if m:
        first = t[:m.start()].rstrip(" ,;:-")

        if first:
            if not first.endswith("?"):
                first += "?"
            return first

    # Two explicitly separated questions.
    qpos = t.find("?")

    if qpos >= 0:
        return t[:qpos + 1].strip()

    # Defensive fallback.
    return t.strip()

def _first_sentence(text: str) -> str:
    """Return the first textual sentence, with pipe delimiters removed."""
    import re
    cleaned = re.sub(r"\|+", " ", text or "").strip()
    if not cleaned:
        return ""
    m = re.search(r"[.!?](?:\s|$)", cleaned)
    return cleaned[:m.end()].strip() if m else cleaned


def _malformed_separator_present(text: str) -> bool:
    """True for pipe runs other than the exact delimiter '|||'."""
    import re
    return any(run != SEP for run in re.findall(r"\|+", text or ""))


def _normalise_to_one_final_separator(text: str) -> str:
    """
    Return exactly one canonical final ||| delimiter.

    RQ3 repair rule:
    specialist prompts prohibit pipe characters anywhere except the
    final delimiter. Therefore, if any pipe is present, everything
    from the first pipe onward is treated as a malformed delimiter
    tail and discarded before appending canonical |||.

    Examples:
        "Question? |"       -> "Question? |||"
        "Question? |'|"     -> "Question? |||"
        'Question? | "|"|'  -> "Question? |||"
        "Question? ||r"     -> "Question? |||"
        "Question? |||"     -> "Question? |||"
    """
    t = _strip_outer_quotes((text or "").strip())

    if not t:
        return SEP

    # Pipes are forbidden in legitimate response content.
    # The first pipe therefore marks the malformed separator tail.
    if "|" in t:
        t = t.split("|", 1)[0].rstrip()

    # Remove quote/backtick debris immediately before where the
    # malformed separator appeared.
    t = t.rstrip(" \\t\\r\\n\\\"'`")

    if not t:
        return SEP

    return f"{t} {SEP}"

def _is_off_topic_response(response: str) -> bool:
    """Cheap response-only screen before the optional semantic safety call.

    The old code concatenated the candidate message with the robot response.
    That meant an unrelated robot response could pass merely because the user's
    answer contained 'python' or another allowed term.
    """
    low = (response or "").lower()
    return not any(keyword in low for keyword in _ALLOWED_TOPICS)


def safety_agent(state: InterviewState) -> InterviewState:
    """Validate, repair and log the specialist response before speech.

    Violation labels describe the RAW specialist output. Repairs are applied
    only after those failures are measured, so RQ3 is not contaminated by a
    repair that creates a second artificial violation.
    """
    raw = (state.get("response", "") or "").strip()
    original_response = raw
    candidate = _strip_outer_quotes(raw)
    last_user = state.get("last_user_message", "")
    agent_ran = state.get("next_agent", "unknown")
    safety_log = list(state.get("safety_log", []))
    violations = []

    expected_questions = 0 if agent_ran == "feedback" else 1

    # ---------- Measure failures on the original specialist output ----------
    q_units = _question_units(candidate)
    if q_units > expected_questions:
        violations.append("UNEXPECTED_QUESTION" if expected_questions == 0 else "MULTI_QUESTION")
    elif q_units < expected_questions:
        violations.append("MISSING_QUESTION")

    original_word_count = len(__import__("re").sub(r"\|+", " ", candidate).split())
    if original_word_count > _SAFETY_MAX_WORDS:
        violations.append("TOO_LONG")

    sep_count = candidate.count(SEP)
    if _malformed_separator_present(candidate):
        violations.append("MALFORMED_SEP")
    if sep_count == 0:
        violations.append("MISSING_SEP")
    elif sep_count > 1:
        violations.append("EXTRA_SEP")
    if sep_count and not candidate.rstrip().endswith(SEP):
        violations.append("SEP_NOT_FINAL")

    # -------------------------- Apply deterministic repairs -------------------
    response = candidate

    if q_units > expected_questions:
        if expected_questions == 0:
            response = "Thank you for your time today. We will review your responses and be in touch shortly."
        else:
            response = _first_question_unit(response)
    elif q_units < expected_questions:
        response = "Could you tell me more about your relevant experience?"

    if original_word_count > _SAFETY_MAX_WORDS:
        if expected_questions == 1:
            shortened = _first_question_unit(response)
            if shortened:
                response = shortened
        else:
            response = _first_sentence(response)

    # Controller/TTS receives one canonical delimiter regardless of what the
    # model originally emitted.
    response = _normalise_to_one_final_separator(response)

    # Cheap response-only screen first.  This avoids the old false-pass where
    # an unrelated robot response looked on-topic only because the USER message
    # happened to contain 'python' or another allowed keyword.
    if _is_off_topic_response(response):
        SAFETY_PROMPT = """You are a safety checker for a professional Junior Data Scientist interview robot.
Return exactly one word: safe or unsafe.
unsafe means the ROBOT response is harmful, grossly off-topic, or inappropriate for a professional interview.
Be lenient with ordinary conversational wording."""

        verdict = _call_llm(
            SAFETY_PROMPT,
            f"Candidate context: {last_user}\nRobot response: {response}",
            max_tokens=5,
        ).strip().lower()

        if verdict.startswith("unsafe"):
            violations.append("OFF_TOPIC")
            if expected_questions == 0:
                response = f"Thank you for your time today. We will review your responses and be in touch shortly. {SEP}"
            else:
                response = f"Could you tell me more about your professional experience? {SEP}"

    violations = list(dict.fromkeys(violations))

    if violations:
        import logging
        log = logging.getLogger("safety_agent")
        log.warning(
            f"[SAFETY] Agent={agent_ran} | Violations={violations} | "
            f"Original='{original_response[:100]}' | Fixed='{response[:100]}'"
        )

    safety_log.append({
        "agent": agent_ran,
        "topic": state.get("current_topic", ""),
        "violations": violations,
        "original": original_response,
        "fixed": response,
    })

    return {**state, "response": response, "safety_log": safety_log}


