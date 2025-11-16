from langgraph.graph import StateGraph, END, START
from state import SupportAgentState
from nodes import tier_classifier, query_issue_classifier, classify_issue, pick_policy, resolve_issue
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

# def should_continue_to_classify(state: SupportAgentState):
#     return "classify" if state.is_support_ticket else END

workflow = StateGraph(SupportAgentState)
#add triagenode(L1L2L3)(Tier Classifier)
workflow.add_node("tier_classification", tier_classifier)#iterupt if L3
workflow.add_node("query_issue_classification", query_issue_classifier)
#Refactor Support Ticket Classification into query/issue classifier
workflow.add_node("classify", classify_issue)
workflow.add_node("policy", pick_policy)
workflow.add_node("resolve", resolve_issue)
#add email responder node
# workflow.add_node("responder", email_responder)
workflow.set_entry_point("tier_classification")
workflow.add_edge(START, "tier_classification")
workflow.add_edge("tier_classification", "query_issue_classification")
workflow.add_edge("query_issue_classification", "classify")
workflow.add_edge("classify", "policy")
#add interupt within the resolve node before the L3 tool calls
workflow.add_edge("policy", "resolve")
# workflow.add_edge("responder", END)
workflow.add_edge("resolve", END)

graph_app = workflow.compile()