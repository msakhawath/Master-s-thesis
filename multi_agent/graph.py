# =============================================================
# multi_agent/graph.py  —  The LangGraph Wiring
# =============================================================
# Connects all FIVE agents into a working graph.
#
# Updated flow (Safety agent added after every specialist):
#
#   START
#     ↓
#   [Orchestrator]  → decides who speaks
#     ├─ "interviewer" → [Interviewer] → [Safety] → END
#     ├─ "prober"      → [Prober]      → [Safety] → END
#     └─ "feedback"    → [Feedback]    → [Safety] → END
#
# Safety is always the LAST agent before the robot speaks.
# It checks and corrects the response, and logs any violations.
# =============================================================

from langgraph.graph import StateGraph, END

from multi_agent.state  import InterviewState
from multi_agent.agents import (
    orchestrator_agent,
    interviewer_agent,
    prober_agent,
    feedback_agent,
    safety_agent,         # ← NEW
)


def _route_from_orchestrator(state: InterviewState) -> str:
    """
    Routing function — reads state["next_agent"] and returns
    the name of the node to go to next.
    """
    return state.get("next_agent", "interviewer")


def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph interview graph with
    the Safety/Policy agent as the final gate before END.
    """

    graph = StateGraph(InterviewState)

    # ── Register all five agents as nodes ───────────────────
    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("interviewer",  interviewer_agent)
    graph.add_node("prober",       prober_agent)
    graph.add_node("feedback",     feedback_agent)
    graph.add_node("safety",       safety_agent)      # ← NEW

    # ── Entry point: always start with Orchestrator ─────────
    graph.set_entry_point("orchestrator")

    # ── Orchestrator routes to one of three specialists ─────
    graph.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {
            "interviewer": "interviewer",
            "prober":       "prober",
            "feedback":     "feedback",
        }
    )

    # ── Every specialist feeds into Safety before END ────────
    # This guarantees EVERY response is checked, regardless
    # of which agent generated it.
    graph.add_edge("interviewer", "safety")
    graph.add_edge("prober",      "safety")
    graph.add_edge("feedback",    "safety")

    # ── Safety is always the last step before END ────────────
    graph.add_edge("safety", END)

    return graph.compile()


# Build once on import — reused every turn
interview_graph = build_graph()
