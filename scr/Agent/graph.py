import os
import sys

# When running the script directly, ensure repo root is on sys.path so
# absolute package imports like `scr` work.
if __name__ == "__main__" and __package__ is None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from scr.Agent.state import SupportAgentState
from scr.Agent.nodes import classify_issue, pick_policy, resolve_issue


workflow = StateGraph(SupportAgentState)
workflow.add_node("classify", classify_issue)
workflow.add_node("policy", pick_policy)
workflow.add_node("resolve", resolve_issue)
workflow.set_entry_point("classify")
workflow.add_edge("classify", "policy")
workflow.add_edge("policy", "resolve")
workflow.add_edge("resolve", END)

graph_app = workflow.compile()


def state_to_mapping(state_obj):
    """Convert a Pydantic model to a plain mapping (supports v2 and v1).

    Returns a dict. If the model_dump()/dict() returns a one-element list
    containing a dict, normalize to that dict.
    """
    if hasattr(state_obj, "model_dump"):
        mapping = state_obj.model_dump()
    elif hasattr(state_obj, "dict"):
        mapping = state_obj.dict()
    else:
        raise TypeError("Unable to convert state object to mapping")

    # Normalize accidental list-wrapped mapping
    if isinstance(mapping, list) and len(mapping) == 1 and isinstance(mapping[0], dict):
        return mapping[0]
    if not isinstance(mapping, dict):
        raise TypeError("Expected mapping (dict) after conversion, got: %r" % type(mapping))
    return mapping


if __name__ == "__main__":
    msg = input("Enter issue description: ")
    # Create a Pydantic state instance and convert to dict for langgraph
    initial_state = SupportAgentState(messages=[{"type": "human", "content": msg}])
    mapping = state_to_mapping(initial_state)

    # langgraph expects a mapping (dict), not a list containing a dict
    final_state = graph_app.invoke(mapping)
    print("Final State:", final_state)