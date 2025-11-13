from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field
from state import SupportAgentState
from tools import check_order_status, track_order, check_stock, initialize_resend, initialize_refund
from langchain_core.tools import tool as create_tool
import json
from typing import Dict, Any, List, Optional
from database.policies import format_policies_for_llm, get_policies_for_problem
from dotenv import load_dotenv
load_dotenv()
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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def support_classification(state: SupportAgentState):     
    issue_text = state.messages[0].content
    
    prompt = (
        """You are a binary classifier. Given a single user message, decide if it is a customer support ticket about orders, shipments, refunds/returns/exchanges, resends/reshipments, stock inquiries, cancellations, payment disputes related to a specific order/product, or any direct request to act on an order (e.g., "check order status", "track order", "initiate refund", "resend item", "check stock", "cancel order", "return item"). 

        Output requirements:
        - Output exactly one token: either True or False (capitalized).
        - No extra text, punctuation, explanation, or formatting.

        Definitions of True:
        - Explicit requests about an order (order IDs, product IDs), shipment tracking, initiating refunds/returns/resends, checking stock for fulfilling an order, cancellations, payment/refund status, or asking the service to perform such actions.
        - Customer complaints about a specific order that request resolution (e.g., wrong/damaged item, missing items, late delivery).

        Definitions of False:
        - General questions (account settings, pricing, working hours), feedback/praise, marketing/feature requests, unrelated conversation, or informational queries not asking for action on an order.

        Examples:
        Input: "What's the status of order 12345?" 
        Output: True

        Input: "Please initiate a refund for order 98765, product SKU: ABC-1"
        Output: True

        Input: "My package tracking number 1Z999AA10123456784 shows delayed — please update" 
        Output: True

        Input: "Is product SKU-432 in stock?" 
        Output: True

        Input: "I received the wrong item and want to return it" 
        Output: True

        Input: "How do I reset my password?" 
        Output: False

        Input: "Do you offer bulk discounts?" 
        Output: False

        Input: "Feature request: add two-factor auth" 
        Output: False

        Now classify the following message and output only True or False:
        <Message to classify>
        ```You are a binary classifier. Given a single user message, decide if it is a customer support ticket about orders, shipments, refunds/returns/exchanges, resends/reshipments, stock inquiries, cancellations, payment disputes related to a specific order/product, or any direct request to act on an order (e.g., "check order status", "track order", "initiate refund", "resend item", "check stock", "cancel order", "return item"). 

        Output requirements:
        - Output exactly one token: either True or False (capitalized).
        - No extra text, punctuation, explanation, or formatting.

        Definitions of True:
        - Explicit requests about an order (order IDs, product IDs), shipment tracking, initiating refunds/returns/resends, checking stock for fulfilling an order, cancellations, payment/refund status, or asking the service to perform such actions.
        - Customer complaints about a specific order that request resolution (e.g., wrong/damaged item, missing items, late delivery).

        Definitions of False:
        - General questions (account settings, pricing, working hours), feedback/praise, marketing/feature requests, unrelated conversation, or informational queries not asking for action on an order.

        Examples:
        Input: "What's the status of order 12345?" 
        Output: True

        Input: "Please initiate a refund for order 98765, product SKU: ABC-1"
        Output: True

        Input: "My package tracking number 1Z999AA10123456784 shows delayed — please update" 
        Output: True

        Input: "Is product SKU-432 in stock?" 
        Output: True

        Input: "I received the wrong item and want to return it" 
        Output: True

        Input: "How do I reset my password?" 
        Output: False

        Input: "Do you offer bulk discounts?" 
        Output: False

        Input: "Feature request: add two-factor auth" 
        Output: False

        Now classify the following message and output only True or False:
        <Message to classify>""")
    
    response = llm.invoke([HumanMessage(content=f"{prompt}\nCustomer issue: {issue_text}")])

    is_ticket = response.content.strip().lower() == ("true" or "yes")
    
    # Add classification message to the conversation
    classification_message = AIMessage(content=f"📁 *Support Ticket Classification*: {'Support Ticket' if is_ticket else 'General Inquiry'}")
    
    return {
        "messages": [*state.messages, classification_message],
        "is_support_ticket": is_ticket
    }



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
    # Create callback handler to capture reasoning
    reasoning_handler = ReasoningCaptureHandler()

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
    
    # Create task description
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
        f"Investigate and resolve this issue step by step."
    )
    
    # Use LLM with tools directly (compatible approach)
    tools = [check_order_status, track_order, check_stock, initialize_resend, initialize_refund]
    llm_with_tools = llm.bind_tools(tools)
    
    # Get response with tool calls
    response = llm_with_tools.invoke([HumanMessage(content=task)])
    
    # Process tool calls if any
    tool_messages = []
    detailed_reasoning = []
    result_text = ""
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_input = tool_call['args']
            
            # Execute the tool
            tool_func = None
            for t in tools:
                if t.name == tool_name:
                    tool_func = t
                    break
            
            if tool_func:
                # Call the tool
                tool_result = tool_func.invoke(tool_input)
                
                # Record reasoning
                detailed_reasoning.append({
                    "thought": f"Calling {tool_name}",
                    "action": tool_name,
                    "action_input": str(tool_input),
                    "result": tool_result
                })
                
                # Add messages
                tool_messages.append(AIMessage(content=f"🤔 Calling {tool_name} with {tool_input}"))
                tool_messages.append(ToolMessage(
                    name=tool_name,
                    content=str(tool_input),
                    tool_call_id=tool_call.get('id', f"call_{len(tool_messages)}")
                ))
                tool_messages.append(AIMessage(content=f"📊 Tool response:\n{tool_result}"))
    
    # Get final response from LLM
    if response.content:
        result_text = response.content
    else:
        # Ask LLM to summarize based on tool results
        summary_prompt = f"Based on the investigation, provide a resolution for the customer.\n\nTask: {task}\n\nTool results: {detailed_reasoning}"
        final_response = llm.invoke([HumanMessage(content=summary_prompt)])
        result_text = final_response.content
    
    # Determine action and reason based on the result
    if "refund" in result_text.lower():
        action = "Refund issued"
        if "stock" in result_text.lower() and ("0" in result_text or "not available" in result_text.lower() or "unavailable" in result_text.lower()):
            reason = "Stock not available for replacement."
        else:
            reason = "Per company policy for this issue type."
    else:
        action = "Resend item"
        reason = "Item in stock and eligible for replacement per policy."

    # Create a summary of the reasoning process
    reasoning_summary = "\n".join([f"Step {i+1}: {step.get('thought', '')}" for i, step in enumerate(detailed_reasoning)])

    # Final resolution message
    resolution_message = AIMessage(content=f"✅ **Resolution**: {action} | Reason: {reason}\n\n{result_text}")

    # Format reasoning for frontend display
    formatted_reasoning = detailed_reasoning

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
