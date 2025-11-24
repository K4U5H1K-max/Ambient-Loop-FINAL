from langgraph.graph import StateGraph, END, START
from state import SupportAgentState
from nodes import validate_and_load_context, tier_classifier, query_issue_classifier, classify_issue, pick_policy, resolve_issue
from database.memory import (
    get_policy_memory,
    seed_policy_memory,
    get_product_memory,
    seed_product_memory,
)
from database.ticket_db import (
    SessionLocal,
    create_tables,
)
from database.seed_policies import seed_policies_from_py
from database.seed_products import seed_products

db = SessionLocal()

try:
    create_tables()
    seed_policies_from_py()
    seed_products()
except Exception as e:
    print(f"[graph.py] DB setup/seeding warning: {e}")
    
try:
    with get_policy_memory() as policy_store:
        seed_policy_memory(db, policy_store)
    with get_product_memory() as product_store:
        seed_product_memory(db, product_store)
except Exception as e:
    print(f"[graph.py] Store seeding warning: {e}")

def route_after_validation(state: SupportAgentState):
    """Route after validation: if support ticket, go to tier classifier, else end."""
    return "tier_classification" if state.is_support_ticket else END

def should_continue_from_tier(state: SupportAgentState):
    """Check if tier classification was approved. If denied, end the flow."""
    return "query_issue_classification" if getattr(state, 'approved', False) else END

def should_continue_to_classify(state: SupportAgentState):
    return "classify" if state.is_support_ticket else END

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
workflow.add_conditional_edges(
    "tier_classification",
    should_continue_from_tier,
    {
        "query_issue_classification": "query_issue_classification",
        END: END
    })
workflow.add_conditional_edges(
    "query_issue_classification",
    should_continue_to_classify,
    {
        "classify": "classify",
        END: END
    }
)
workflow.add_edge("classify", "policy")
workflow.add_edge("policy", "resolve")
workflow.add_edge("resolve", END)


graph_app = workflow.compile()

