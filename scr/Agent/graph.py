from langgraph.graph import StateGraph, END, START
from state import SupportAgentState
from nodes import classify_issue, pick_policy, resolve_issue, support_classification
from langgraph.checkpoint.memory import MemorySaver


def should_continue_to_classify(state: SupportAgentState):
    return "classify" if state.is_support_ticket else END

workflow = StateGraph(SupportAgentState)
workflow.add_node("support_ticket_classification", support_classification)
workflow.add_node("classify", classify_issue)
workflow.add_node("policy", pick_policy)
workflow.add_node("resolve", resolve_issue)
# workflow.add_node("responder", email_responder)
workflow.set_entry_point("support_ticket_classification")
workflow.add_conditional_edges(
    "support_ticket_classification",
    should_continue_to_classify,
    {
        "classify": "classify",
        END: END
    }
)
workflow.add_edge("classify", "policy")
workflow.add_edge("policy", "resolve")
# workflow.add_edge("responder", END)
workflow.add_edge("resolve", END)

graph_app = workflow.compile()