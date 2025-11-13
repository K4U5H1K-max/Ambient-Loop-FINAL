from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field
from state import SupportAgentState
from tools import check_order_status, track_order, check_stock, initialize_resend, initialize_refund
from langchain.agents import Tool, initialize_agent, AgentType
import json
from typing import Dict, Any, List, Optional
from policies import format_policies_for_llm, get_policies_for_problem



from langgraph.graph import StateGraph, END, START

# Custom callback handler to capture agent reasoning
class ReasoningCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        self.reasoning_steps = []
        self.current_step = {}
    
    def on_agent_action(self, action, **kwargs):
        self.current_step = {
            "action": action.tool,
            "action_input": action.tool_input,
            "thought": action.log
        }
        self.reasoning_steps.append(self.current_step)
        
    def on_tool_end(self, output, **kwargs):
        if self.current_step:
            self.current_step["tool_output"] = output
            
    def get_reasoning(self):
        return self.reasoning_steps

# Define Pydantic models for structured outputs
class IssueClassification(BaseModel):
    problem_types: List[str] = Field(description="List of identified problem types")
    reasoning: str = Field(description="Detailed reasoning for the classification")

class PolicySelection(BaseModel):
    policy_name: str = Field(description="Name of the selected policy from the provided policy list")
    policy_description: str = Field(description="Description of the selected policy")
    reasoning: str = Field(description="Detailed reasoning for selecting this policy based on the customer issue and problem types")
    application_notes: Optional[str] = Field(description="Specific notes on how to apply this policy to the current situation", default=None)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

def classify_issue(state: SupportAgentState):
    prompt = (
        "You are a customer support AI Agent. Analyze the following customer issue and identify the problem types.\n"
        "Select from the following categories:\n"
        "- non-delivery: Customer hasn't received their order\n"
        "- delayed: Order is taking longer than expected\n"
        "- damaged: Product arrived damaged or defective\n"
        "- wrong-item: Customer received incorrect product\n"
        "- quality: Product quality didn't meet expectations\n"
        "- fit: Size or fit issue with clothing/wearable\n"
        "- return: Customer wants to return an item\n"
        "- refund: Customer is requesting a refund\n"
        "- account: Issues with customer's account\n"
        "- website: Problems with the website\n"
        "- general: Any other general inquiries\n"
    )
    
    issue_text = state.messages[0].content
    
    # Create structured output LLM
    structured_llm = llm.with_structured_output(IssueClassification)
    
    # Get structured response
    response = structured_llm.invoke(
        [HumanMessage(content=f"{prompt}\nCustomer issue: {issue_text}")]
    )
    
    # Extract data from structured response
    problems = response.problem_types
    reasoning = response.reasoning
    
    # Format the problems for display
    problem_display = ", ".join([f"`{p}`" for p in problems])
    
    # Add analysis message to the conversation
    analysis_message = AIMessage(content=f"🔎 **Issue Analysis**:\n{reasoning}")
    
    # Add classification message to the conversation
    classification_message = AIMessage(content=f"📁 **Identified Problem Types**: {problem_display}")
    
    return {
        "messages": [*state.messages, analysis_message, classification_message],
        "problems": problems,
        "reasoning": {"classify": reasoning},
        "thought_process": state.thought_process + [{
            "step": "classify_issue",
            "reasoning": reasoning,
            "output": ", ".join(problems)
        }]
    }

def pick_policy(state: SupportAgentState):
    issue_text = state.messages[0].content  # Original customer message
    problems_str = ", ".join(state.problems)
    
    # Get all relevant policies based on the identified problems
    relevant_policies = {}
    for problem in state.problems:
        problem_policies = get_policies_for_problem(problem)
        relevant_policies.update(problem_policies)
    
    # If no specific policies found, get all policies
    if not relevant_policies:
        all_policies = format_policies_for_llm()
        policies_text = all_policies
    else:
        # Format the relevant policies for the LLM
        policies_text = "# Relevant Customer Support Policies\n\n"
        for name, policy in relevant_policies.items():
            policies_text += f"## {name}\n"
            policies_text += f"Description: {policy['description']}\n"
            policies_text += f"When to use: {policy['when_to_use']}\n"
            policies_text += f"Applicable problems: {', '.join(policy['applicable_problems'])}\n\n"
    
    # Get the issue classification reasoning to provide context
    classification_reasoning = state.reasoning.get("classify", "")
    
    prompt = (
        "You are a support AI. Based on the customer issue and identified problem types, "
        "determine the most appropriate company policy to apply from the provided list.\n\n"
        "Review the policies carefully and select the one that best addresses the customer's situation.\n"
        "Explain your reasoning for the policy selection and provide specific notes on how to apply it."
    )
    
    # Create structured output LLM
    structured_llm = llm.with_structured_output(PolicySelection)
    
    # Get structured response
    response = structured_llm.invoke([
        HumanMessage(content=f"{prompt}\n\nCustomer Issue: {issue_text}\n\n"
                            f"Problem Types: {problems_str}\n\n"
                            f"Issue Analysis: {classification_reasoning}\n\n"
                            f"Available Policies:\n{policies_text}")
    ])
    
    # Extract data from structured response
    policy_name = response.policy_name
    policy_desc = response.policy_description
    reasoning = response.reasoning
    application_notes = response.application_notes or ""
    
    # Add reasoning message to the conversation
    reasoning_message = AIMessage(content=f"🔍 **Policy Analysis**:\n{reasoning}")
    
    # Add policy selection message to the conversation
    policy_content = f"📋 **Selected Policy**: {policy_name}\n{policy_desc}"
    if application_notes:
        policy_content += f"\n\n📝 **Application Notes**: {application_notes}"
    policy_message = AIMessage(content=policy_content)
        
    return {
        "messages": [*state.messages, reasoning_message, policy_message],
        "policy_name": policy_name,
        "policy_desc": policy_desc,
        "reasoning": {**state.reasoning, "policy": reasoning},
        "thought_process": state.thought_process + [{
            "step": "pick_policy",
            "reasoning": reasoning,
            "output": f"{policy_name}: {policy_desc}"
        }]
    }

def resolve_issue(state: SupportAgentState):
    tools = [
        Tool(
            name="check_order_status", 
            func=check_order_status, 
            description="Check order status by order ID. Example: ORD12345"
        ),
        Tool(
            name="track_order", 
            func=track_order, 
            description="Track shipment by order ID. Example: ORD12345"
        ),
        Tool(
            name="check_stock", 
            func=check_stock, 
            description="Check item stock availability by product ID. Product IDs are P1001 (Premium Wireless Headphones), P1002 (Smart Fitness Watch), P1003 (Organic Cotton T-Shirt), P1004 (Stainless Steel Water Bottle), P1005 (Wireless Charging Pad)."
        ),
        Tool(
            name="initialize_resend", 
            func=initialize_resend, 
            description="Resend item to customer. Format: 'ORD12345/P1001' where ORD12345 is the order ID and P1001 is the product ID."
        ),
        Tool(
            name="initialize_refund", 
            func=initialize_refund, 
            description="Refund customer for order. Format: 'ORD12345/P1001' where ORD12345 is the order ID and P1001 is the product ID."
        )
    ]
    
    # Create callback handler to capture reasoning
    reasoning_handler = ReasoningCaptureHandler()
    
    # Initialize agent with callback handler
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        callbacks=[reasoning_handler]
    )

    issue_text = state.messages[0].content
    policy_info = f"{state.policy_name}: {state.policy_desc}"
    problems_str = ", ".join(state.problems)
    
    # Product ID mapping for reference
    product_mapping = """
    Product ID Reference:
    - P1001: Premium Wireless Headphones ($199.99)
    - P1002: Smart Fitness Watch ($149.99)
    - P1003: Organic Cotton T-Shirt ($29.99)
    - P1004: Stainless Steel Water Bottle ($34.99)
    - P1005: Wireless Charging Pad ($39.99)
    """
    
    task = (
        f"You are a customer support agent handling the following issue:\n"
        f"Customer issue: {issue_text}\n"
        f"Identified problem types: {problems_str}\n"
        f"Company policy: {policy_info}\n\n"
        f"{product_mapping}\n"
        f"Follow these guidelines:\n"
        f"1. First, extract the order ID from the customer issue (format: ORD#####)\n"
        f"2. For non-delivery issues:\n"
        f"   - Check order status using check_order_status\n"
        f"   - Check tracking information using track_order\n"
        f"3. For damaged or defective product issues:\n"
        f"   - Identify the product from the customer's message\n"
        f"   - Check stock availability using check_stock\n"
        f"   - If stock is available, initiate a resend using initialize_resend\n"
        f"   - If stock is not available (level 0), initiate a refund using initialize_refund\n"
        f"4. For wrong item issues:\n"
        f"   - Identify both the incorrect item received and the correct item ordered\n"
        f"   - Check stock of correct item using check_stock\n"
        f"   - If correct item is in stock, initiate a resend using initialize_resend\n"
        f"   - If correct item is out of stock, initiate a refund using initialize_refund\n"
        f"5. For any other issues: Apply the relevant policy\n\n"
        f"Use the available tools to investigate and resolve this issue. Explain your reasoning step by step."
    )
    
    result = agent.run(task)
    
    # Get detailed reasoning from callback handler
    detailed_reasoning = reasoning_handler.get_reasoning()
    
    # Format reasoning for frontend display
    formatted_reasoning = []
    
    # Create messages for each tool call and response
    tool_messages = []
    for step in detailed_reasoning:
        # Format step for the detailed reasoning
        formatted_step = {
            "thought": step.get("thought", ""),
            "action": step.get("action", ""),
            "action_input": step.get("action_input", ""),
            "result": step.get("tool_output", "")
        }
        formatted_reasoning.append(formatted_step)
        
        # Add tool thought as AI message
        if step.get("thought"):
            tool_messages.append(AIMessage(content=f"🤔 {step.get('thought')}"))
        
        # Add tool call as a ToolMessage
        if step.get("action") and step.get("action_input"):
            tool_name = step.get("action")
            tool_input = step.get("action_input")
            
            # Create proper tool message with name and input
            tool_messages.append(ToolMessage(
                name=tool_name,
                content=tool_input,
                tool_call_id=f"call_{len(tool_messages)}"
            ))
        
        # Add tool response as a separate message
        if step.get("tool_output"):
            tool_messages.append(AIMessage(
                content=f"📊 Tool response:\n{step.get('tool_output')}"
            ))
    
    # Determine action and reason based on the result
    if "refund" in result.lower():
        action = "Refund issued"
        if "stock" in result.lower() and ("0" in result or "not available" in result.lower() or "unavailable" in result.lower()):
            reason = "Stock not available for replacement."
        else:
            reason = "Per company policy for this issue type."
    else:
        action = "Resend item"
        reason = "Item in stock and eligible for replacement per policy."

    # Create a summary of the reasoning process
    reasoning_summary = "\n".join([f"Step {i+1}: {step.get('thought', '')}" for i, step in enumerate(detailed_reasoning)])

    # Final resolution message
    resolution_message = AIMessage(content=f"✅ **Resolution**: {action} | Reason: {reason}\n\n{result}")

    return {
        "messages": [*state.messages, *tool_messages, resolution_message],
        "action_taken": action,
        "reason": reason,
        "reasoning": {**state.reasoning, "resolve": reasoning_summary},
        "thought_process": state.thought_process + [{
            "step": "resolve_issue",
            "reasoning": reasoning_summary,
            "detailed_steps": formatted_reasoning,
            "output": f"{action} - {reason}"
        }]
    }





def create_pipeline() -> StateGraph:
    from langgraph.graph import StateGraph, END
    workflow = StateGraph(SupportAgentState)
    workflow.add_node("classify", classify_issue)
    workflow.add_node("policy", pick_policy)
    workflow.add_node("resolve", resolve_issue)
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "policy")
    workflow.add_edge("policy", "resolve")
    workflow.add_edge("resolve", END)
    return workflow
def process_email_through_pipeline(email_text: str) -> dict:
    """
    Runs the pipeline for a given email text and returns the final state.
    """
    workflow = create_pipeline()
    g = workflow.compile()

    initial_state = {
        "messages": email_text,
        "problems": [],
        "policy_name": "",
        "policy_desc": "",
        "policy_reason": "",
        "action_taken": "",
        "reason": "",
        "reasoning": {},
        "thought_process": []
    }
    result = g.invoke(initial_state)
    return result