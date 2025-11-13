from langgraph.graph import StateGraph, END, START
from state import SupportAgentState
from nodes import classify_issue, pick_policy, resolve_issue
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(SupportAgentState)
workflow.add_node("classify", classify_issue)
workflow.add_node("policy", pick_policy)
workflow.add_node("resolve", resolve_issue)
workflow.set_entry_point("classify")
workflow.add_edge("classify", "policy")
workflow.add_edge("policy", "resolve")
workflow.add_edge("resolve", END)

graph_app = workflow.compile()