from typing import List, Annotated, Dict, Optional, Any
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SupportAgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []
    
    # Ticket classification
    is_support_ticket: bool = False
    order_id: Optional[str] = None
    has_valid_order_id: bool = False
    
    # Issue classification
    problems: List[str] = []
    query_issue: str = ""
    
    # Tier classification
    tier_level: str = ""
    approved: Optional[bool] = None
    
    # Policy details
    policy_name: str = ""
    policy_desc: str = ""
    policy_reason: str = ""
    
    # Resolution
    action_taken: str = ""
    reason: str = ""
    email_reply: Optional[str] = None
    requires_human_review: bool = False

    # Reasoning tracking
    reasoning: Dict[str, str] = {}
    thought_process: List[Dict[str, Any]] = []