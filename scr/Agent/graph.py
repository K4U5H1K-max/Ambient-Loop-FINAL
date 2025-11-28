from langgraph.graph import StateGraph, END, START
from state import SupportAgentState
from nodes import (
    validate_and_load_context,
    tier_classifier,
    query_issue_classifier,
    classify_issue,
    pick_policy,
    resolve_issue,
)

from database.memory import (
    get_policy_memory,
    seed_policy_memory,
    get_product_memory,
    seed_product_memory,
)
# Database session removed - seeding handled by init_db.py

def route_after_validation(state: SupportAgentState):
    return "tier_classification" if state.is_support_ticket and state.has_order_id else END

workflow = StateGraph(SupportAgentState)

workflow.add_node("validate", validate_and_load_context)
workflow.add_node("tier_classification", tier_classifier)
workflow.add_node("query_issue_classification", query_issue_classifier)
workflow.add_node("classify", classify_issue)
workflow.add_node("policy", pick_policy)
workflow.add_node("resolve", resolve_issue)

workflow.set_entry_point("validate")

workflow.add_conditional_edges(
    "validate",
    route_after_validation,
    {
        "tier_classification": "tier_classification",
        END: END
    }
)

workflow.add_edge("tier_classification", "query_issue_classification")
workflow.add_edge("query_issue_classification","classify")
workflow.add_edge("classify", "policy")
workflow.add_edge("policy", "resolve")

# Final step: resolve -> END
workflow.add_edge("resolve", END)

graph_app = workflow.compile()