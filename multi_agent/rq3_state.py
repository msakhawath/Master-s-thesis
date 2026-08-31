# =============================================================
# multi_agent/state.py  —  The Shared Notepad
# =============================================================
# Think of this file as defining a shared notepad that every
# agent in the system can read and write on.
#
# In LangGraph, this is called the "State". Every time a user
# speaks, the state gets updated. Every agent reads from it
# and writes its decision back into it.
# =============================================================

from typing import TypedDict, List, Dict, Optional

# The list of interview topics — in order.
# The Orchestrator works through these one by one.
INTERVIEW_TOPICS = [
    "education and academic background",
    "programming skills and tools",
    "machine learning and modelling experience",
    "data wrangling and preprocessing",
    "model evaluation and metrics",
    "data visualization and storytelling",
    "real-world data science projects",
    "teamwork and communication",
    "problem solving approach",
    "career goals in data science",
]

class InterviewState(TypedDict):
    """
    This is the shared notepad.
    Every agent reads this and every agent writes back to it.

    Think of it like a shared Google Doc that the whole team
    can see and edit during the interview.
    """

    # ── Conversation memory ──────────────────────────────────
    # The full back-and-forth conversation so far.
    # Each entry is {"role": "user"/"assistant", "content": "..."}
    history: List[Dict[str, str]]

    # The most recent thing the user said (shortcut — saves
    # re-reading the whole history every time)
    last_user_message: str

    # ── Interview progress ───────────────────────────────────
    # Which topic are we currently discussing?
    # e.g. "teamwork", "Agile methodology"
    current_topic: str

    # List of topics we have already asked about.
    # The Orchestrator uses this to know what's left.
    topics_covered: List[str]

    # Has the Prober already asked a follow-up on the current topic?
    # We only probe ONCE per topic to avoid annoying the candidate.
    probed_current_topic: bool

    # Which phase is the interview in?
    # "opening"  → greeting, warm-up
    # "main"     → asking topic questions
    # "closing"  → all topics done, give feedback
    interview_phase: str

    # ── Agent routing ────────────────────────────────────────
    # The Orchestrator writes this to tell the graph which
    # agent should speak next.
    # Possible values: "interviewer", "prober", "feedback"
    next_agent: str

    # ── Output ──────────────────────────────────────────────
    # The final text response that will be sent to the Furhat robot.
    # This is filled in by whichever agent runs.
    response: str

    # ── Safety log (for RQ3 — failure mode analysis) ────────
    # Every turn, the Safety/Policy agent appends one entry here.
    # At the end of the session, this gives you a complete record
    # of every violation caught, which agent caused it, and what
    # the original and fixed responses looked like.
    # This data directly answers RQ3 in your thesis.
    # RQ3 experimental condition: "full", "no_safety", or "no_persona"
    rq3_condition: str

    safety_log: List[Dict]
