from langgraph.graph import StateGraph, END, START
from state import SupportAgentState
from nodes import classify_issue, pick_policy, resolve_issue, support_classification
from langgraph.checkpoint.memory import MemorySaver
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

# Ensure relational tables exist and seed them from code (idempotent)
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