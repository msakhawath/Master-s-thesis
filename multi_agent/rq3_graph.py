from langgraph.graph import StateGraph, END

from multi_agent.rq3_state import InterviewState
from multi_agent.rq3_agents import (
    orchestrator_agent,
    interviewer_agent,
    prober_agent,
    feedback_agent,
    safety_agent,
)

VALID_RQ3_CONDITIONS = {
    "full",
    "no_safety",
    "no_persona",
}


def _route_from_orchestrator(state: InterviewState) -> str:
    return state.get("next_agent", "interviewer")


def build_rq3_graph(condition: str = "full"):
    """
    Build one RQ3 experimental graph.

    full:
        Orchestrator -> Specialist -> Safety -> END

    no_safety:
        Orchestrator -> Specialist -> END

    no_persona:
        Same graph as full, but specialist prompts omit
        explicit persona/style wording.
    """

    if condition not in VALID_RQ3_CONDITIONS:
        raise ValueError(
            f"Unknown RQ3 condition: {condition}. "
            f"Expected one of {sorted(VALID_RQ3_CONDITIONS)}"
        )

    graph = StateGraph(InterviewState)

    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("interviewer", interviewer_agent)
    graph.add_node("prober", prober_agent)
    graph.add_node("feedback", feedback_agent)

    use_safety = condition != "no_safety"

    if use_safety:
        graph.add_node("safety", safety_agent)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {
            "interviewer": "interviewer",
            "prober": "prober",
            "feedback": "feedback",
        },
    )

    if use_safety:
        graph.add_edge("interviewer", "safety")
        graph.add_edge("prober", "safety")
        graph.add_edge("feedback", "safety")
        graph.add_edge("safety", END)
    else:
        graph.add_edge("interviewer", END)
        graph.add_edge("prober", END)
        graph.add_edge("feedback", END)

    return graph.compile()
