from typing import List, Annotated, Dict
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SupportAgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []
    is_support_ticket: bool = False
    problems: List[str] = []
    policy_name: str = ""
    policy_desc: str = ""
    policy_reason: str = ""
    action_taken: str = ""
    reason: str = ""
    # Capture reasoning at each step
    reasoning: Dict[str, str] = {}
    # Track agent's thought process
    thought_process: List[Dict[str, str]] = []
