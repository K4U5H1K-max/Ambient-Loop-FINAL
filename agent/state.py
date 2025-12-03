from typing import List, Annotated, Dict, Optional, Any
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SupportAgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []
    #add a issue classifier
    is_support_ticket: bool = False# default to false for support ticket it is going to be over written to true when classified in validation node
    order_id: Optional[str] = None
    products_cache: Optional[str] = None  # Preloaded products context
    problems: List[str] = []
    #query issue classification
    query_issue: str = ""
    #tier classification
    tier_level: str = ""
    # L3 approval
    approved: Optional[bool] = None
    #policy details
    policy_name: str = ""
    policy_desc: str = ""
    policy_reason: str = ""
    action_taken: str = ""
    reason: str = ""
    email_reply: Optional[str] = None

    # Capture reasoning at each step
    reasoning: Dict[str, str] = {}
    # Track agent's thought process
    thought_process: List[Dict[str, Any]] = []
    #Checking for Order ID
    has_order_id: bool = False
    
    # Email metadata for callback (set by mail_api before graph execution)
    gmail_msg_id: Optional[str] = None
    gmail_thread_id: Optional[str] = None
    sender_email: Optional[str] = None
    email_subject: Optional[str] = None
    callback_url: Optional[str] = None