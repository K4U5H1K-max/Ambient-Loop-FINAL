from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from agent.state import SupportAgentState
from agent.tools import check_order_status, track_order, check_stock, initialize_resend, initialize_refund
from langchain_core.tools import tool as create_tool
import json
from typing import Dict, Any, List, Optional
from data.policies import format_policies_for_llm, get_policies_for_problem
from data.data import ORDERS
from difflib import get_close_matches
from dotenv import load_dotenv
load_dotenv()
from data.memory import get_policies_context, get_products_context
import re
from agent.prompts import CLASSIFICATION_PROMPT, TIER_CLASSIFIER_PROMPT, POLICY_SELECTION_PROMPT, ISSUE_CLASSIFIER_PROMPT, RESOLUTION_SUMMARY_PROMPT

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

# Add this near the top with other Pydantic models
class TicketValidation(BaseModel):
    is_support_ticket: bool = Field(description="Whether this is a customer support ticket")
    has_order_id: bool = Field(description="Whether an order ID is present in the message")
    extracted_order_id: Optional[str] = Field(description="The extracted order ID if present (format: ORD#####)", default=None)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def validate_and_load_context(state: SupportAgentState):
    """Validate if message is support ticket and preload products context if yes."""
    print("---VALIDATING TICKET AND LOADING CONTEXT---")
    
    # Get the latest user message
    user_message = None
    for msg in reversed(state.messages):
        if msg.type == "human":
            user_message = msg.content
            break
    
    if not user_message:
        return {"is_support_ticket": False}
    
    # Use structured LLM output
    classification_prompt = CLASSIFICATION_PROMPT.format(user_message=user_message)

    structured_llm = llm.with_structured_output(TicketValidation)
    response = structured_llm.invoke([HumanMessage(content=classification_prompt)])
    
    if not response.is_support_ticket:
        print("Not a support ticket - ending workflow")
        return {"is_support_ticket": False}
    
    # Support ticket confirmed
    if response.has_order_id and response.extracted_order_id:
        order_id_upper = response.extracted_order_id.upper()
        
        # Validate order exists
        if order_id_upper in ORDERS.keys():
            products_context = get_products_context()
            print(f"✅ Valid order ID: {order_id_upper}")
            print(f"Loaded products context: {len(products_context)} chars")
            return {
                "is_support_ticket": True,
                "has_order_id": True,
                "order_id": order_id_upper,
                "products_cache": products_context
            }
        else:
            print(f"⚠️ Order ID {order_id_upper} not found in database")
            return {
                "is_support_ticket": True,
                "has_order_id": False,
                "order_id": None,
                "products_cache": None
            }
    else:
        print("No order ID found in message")
        return {
            "is_support_ticket": True,
            "has_order_id": False,
            "order_id": None,
            "products_cache": None
        }
def tier_classifier(state: SupportAgentState):
    issue_text = state.messages[0].content
    
    prompt = TIER_CLASSIFIER_PROMPT
    
    response = llm.invoke([HumanMessage(content=f"{prompt}\nCustomer issue: {issue_text}")])
    response_text = response.content.strip().lower()

    if "l1" in response_text:
        tier_level = "L1"
    elif "l2" in response_text:
        tier_level = "L2"
    else:
        tier_level = "L3"

    # Interrupt and capture the decision
    # if tier_level == "L3":
    #     decision = interrupt({
    #         "type": "tier_approval",
    #         "tier": tier_level,
    #         "message": f"This issue is classified as {tier_level}. Approve or Deny?",
    #         "options": ["Deny", "Approve"]
    #     })

    # # Process the decision - handle various response formats
    #     approved = decision == "1"
        
    #     # Create message based on approval status
    #     if approved:
    #         status_msg = f"{tier_level} classification approved."
    #     else:
    #         status_msg = f"{tier_level} classification denied."
        
    #     return {
    #         "tier_level": tier_level,
    #         "approved": approved,
    #         "messages": [*state.messages, AIMessage(content=status_msg)]
    #     }
    # else:
    #     return {
    #         "tier_level": tier_level,
    #         "approved": True,
    #         "messages": [*state.messages, AIMessage(content=f"{tier_level} classification automatically approved.")]
    #     }

    if tier_level == "L3":
        # Build interrupt request (SAME FORMAT as the refund/resend one)
        request = {
            "action_request": {
                "action": "tier_classification",
                "args": {
                    "tier": tier_level,
                    "issue": issue_text
                }
            },
            "config": {
                "allow_ignore": True,
                "allow_respond": False,
                "allow_edit": False,
                "allow_accept": True
            },
            "description": f"""
    A Tier 3 classification requires human approval.

    The AI has classified this issue as **{tier_level}** based on complexity and business impact.

    Approve to continue with Tier 3 handling or ignore to downgrade.
    """
        }

        # Send interrupt call and get human decision
        resp = interrupt(request)[0]

        if resp["type"] == "accept":
            approved = True
            status_msg = f"{tier_level} classification approved."
        else:
            approved = False
            status_msg = f"{tier_level} classification denied."

        return {
            "tier_level": tier_level,
            "approved": approved,
            "messages": [*state.messages, AIMessage(content=status_msg)]
        }



#Refactored support ticket classification into query/issue classifier

def query_issue_classifier(state: SupportAgentState):     
    issue_text = state.messages[0].content
    
    prompt = (
        """you are a customer support AI Agent whose primary role is to classify if the incoming customer issue is a support ticket(Issue) or a general inquiry(Query).""")
    
    response = llm.invoke([HumanMessage(content=f"{prompt}\nCustomer issue: {issue_text}")])

    # Parse the response - check if it contains "true" (case-insensitive)
    response_text = response.content.strip().lower()
    answer = "query" if "query" in response_text else "issue"

    #removed this part because the support ticket is default true

    # is_ticket = response_text in ("true", "yes") or "true" in response_text
    
    # # Add classification message to the conversation
    # classification_message = AIMessage(content=f"📁 *Support Ticket Classification*: {'Support Ticket' if is_ticket else 'General Inquiry'}")
    
    return {
        "query_issue": answer
    }



def classify_issue(state: SupportAgentState):
    prompt = ISSUE_CLASSIFIER_PROMPT
    
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
    issue_text = state.messages[0].content
    problems_str = ", ".join(state.problems)
    classification_reasoning = state.reasoning.get("classify", "")

    # Fetch policies from memory store and inject as explicit context
    policies_context = get_policies_context()
    # Try to obtain a concrete list of candidate policies from the policy service
    candidate_policies = None
    try:
        candidate_policies = get_policies_for_problem(issue_text)
    except Exception:
        candidate_policies = None

    # Normalize candidate policy names and descriptions
    allowed_names = []
    policy_map = {}
    if isinstance(candidate_policies, list) and candidate_policies:
        for p in candidate_policies:
            # support dicts with different keys
            name = p.get("name") if isinstance(p, dict) else None
            if not name:
                name = p.get("policy_name") if isinstance(p, dict) else None
            desc = p.get("description") if isinstance(p, dict) else None
            if not name and isinstance(p, str):
                # If entries are strings, try to split into name/desc
                name = p
            if name:
                allowed_names.append(name)
                policy_map[name] = desc or ""

    # If no structured candidates, fall back to the pre-formatted policy text
    policies_text = format_policies_for_llm() if not policies_context else policies_context

    # Build a strict prompt that enumerates candidates (if available) and requires exact selection
    if allowed_names:
        candidates_block = "\n".join([f"{i+1}. {n} - {policy_map.get(n,'')}" for i, n in enumerate(allowed_names)])
        instruction = POLICY_SELECTION_PROMPT.format(
            candidate_policies=candidates_block,
            customer_issue=issue_text,
            problem_types=problems_str,
            issue_analysis=classification_reasoning
        )
    else:
        instruction = POLICY_SELECTION_PROMPT.format(
            candidate_policies="No specific policies found in context.",
            customer_issue=issue_text,
            problem_types=problems_str,
            issue_analysis=classification_reasoning
        )

    structured_llm = llm.with_structured_output(PolicySelection)
    response = structured_llm.invoke([HumanMessage(content=instruction)])

    policy_name = getattr(response, "policy_name", None)
    policy_desc = getattr(response, "policy_description", "")
    reasoning = getattr(response, "reasoning", "")
    application_notes = getattr(response, "application_notes", None) or ""

    # Validate the returned policy_name
    selected_name = None
    if policy_name and allowed_names:
        # direct match
        if policy_name in allowed_names:
            selected_name = policy_name
        else:
            # try fuzzy matching
            matches = get_close_matches(policy_name, allowed_names, n=1, cutoff=0.6)
            if matches:
                selected_name = matches[0]

    # If still no selection, try to pick a safe fallback
    if not selected_name:
        if allowed_names:
            # fallback to the first candidate and record that this was a fallback
            selected_name = allowed_names[0]
            policy_desc = policy_map.get(selected_name, policy_desc or "")
            reasoning = (reasoning or "") + "\n\n(Fallback: model did not return an exact policy name; selected first candidate.)"
        else:
            # No candidates available, use UNKNOWN and keep the model's description
            selected_name = policy_name or "UNKNOWN"

    # Compose messages for the conversation
    reasoning_message = AIMessage(content=f"🔍 **Policy Analysis**:\n{reasoning}")
    policy_content = f"📋 **Selected Policy**: {selected_name}\n{policy_desc}"
    if application_notes:
        policy_content += f"\n\n📝 **Application Notes**: {application_notes}"
    policy_message = AIMessage(content=policy_content)

    # Log selection for auditing (printed output will appear in logs)
    print(f"pick_policy: selected='{selected_name}' (validated: {selected_name in allowed_names}), original_returned='{policy_name}'")

    return {
        "messages": [*state.messages, reasoning_message, policy_message],
        "policy_name": selected_name,
        "policy_desc": policy_desc,
        "reasoning": {**state.reasoning, "policy": reasoning},
        "thought_process": state.thought_process + [{
            "step": "pick_policy",
            "reasoning": reasoning,
            "output": f"{selected_name}: {policy_desc}"
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
    
    # Use cached products context from validation node
    products_context = state.products_cache or ""

    # SHORT-CIRCUIT: if this is a general policy/query request (no actionable order),
    # do NOT call tools. Instead, return a policy-only informational email.
    query_issue_flag = getattr(state, "query_issue", None)
    #problems_lower = ", ".join(state.problems).lower() if getattr(state, "problems", None) else ""
    # if query_issue_flag == "query" or "general" :
    #     policies_context = get_policies_context()
    #     policy_text = state.policy_desc or policies_context or (
    #         "Damaged Item Policy: If a customer receives a damaged or defective item, they are eligible for an immediate replacement or full refund, including shipping costs."
    #     )

        # email_body = (
        #     "Resolution Summary:\n"
        #     "Dear Valued Customer,\n\n"
        #     "Thank you for contacting [Company Name]. We appreciate you reaching out to us.\n\n"
        #     "Regarding our damaged product policy: "
        #     f"{policy_text}\n\n"
        #     "If you would like us to investigate a specific order or request a replacement or refund, please provide your order number (format: ORD#####) and any photos of the damaged item. Once we have that information we can check order status, stock availability, and proceed with a replacement or refund if applicable.\n\n"
        #     "Kind regards,\n"
        #     "Customer Support Team\n"
        #     "[Company Name]"
        # )

        # resolution_message = AIMessage(content=email_body)
        # return {
        #     "messages": [*state.messages, resolution_message],
        #     "action_taken": "Policy Info",
        #     "reason": "Provided policy information without tool calls",
        #     "reasoning": {**state.reasoning, "resolve": "Policy-only response (no tools used)"},
        #     "thought_process": state.thought_process + [{
        #         "step": "resolve_issue",
        #         "reasoning": "Policy-only response",
        #         "output": "Policy Info provided to customer"
        #     }]
        # }

    # Create task description
    task = (
        f"You are a customer support agent handling the following issue:\n"
        f"Customer issue: {issue_text}\n"
        f"Identified problem types: {problems_str}\n"
        f"Company policy: {policy_info}\n\n"
        f"Product Memory Context (from store):\n{products_context}\n\n"
        f"Query/Issue Classification: {query_issue_flag}\n\n"
        f"Has Order ID: {state.has_order_id}\n\n"
        f"Instructions:\n"
        f"1. Extract order ID (format: ORD#####) only if has_order_id is True.\n"
        f"2. Use tools as needed. Do not fabricate data not shown.\n"
        f"3. Choose resend vs refund based strictly on stock availability and policy guidance.\n"
        f"4. Keep reasoning concise but stepwise.\n"
        f"5. If product not found in context, still proceed using tools to validate.\n\n"
        f"Follow these guidelines:\n"
        f"1. First, extract the order ID from the customer issue (format: ORD#####)\n"
        f"2. For non-delivery issues:\n"
        f"   - Check order status using check_order_status_tool\n"
        f"   - Check tracking information using track_order_tool\n"
        f"3. For damaged or defective product issues:\n"
        f"   - Identify the product from the customer's message\n"
        f"   - Check stock availability using check_stock_tool\n"
        f"   - If stock is available, initiate a resend using initialize_resend_tool\n"
        f"   - If stock is not available (level 0), initiate a refund using initialize_refund_tool\n"
        f"4. For wrong item issues:\n"
        f"   - Identify both the incorrect item received and the correct item ordered\n"
        f"   - Check stock of correct item using check_stock_tool\n"
        f"   - If correct item is in stock, initiate a resend using initialize_resend_tool\n"
        f"   - If correct item is out of stock, initiate a refund using initialize_refund_tool\n"
        f"5. For any other issues: Apply the relevant policy\n\n"
        f"IMPORTANT: After completing your investigation, you MUST format your final response as a professional customer support email using this exact structure:\n\n"
        # f"Resolution Summary:\n"
        f"Dear [Customer Name],\n\n"
        f"Thank you for reaching out to us. We appreciate you contacting [Company Name].\n\n"
        f"I understand that you are experiencing [briefly describe their issue], and after carefully reviewing your situation, "
        f"we have identified [what was found] and determined the appropriate resolution. We will be [action being taken] "
        f"to resolve this for you as quickly as possible.\n\n"
        f"[Include specific details: order number, timeline, next steps, tracking info if applicable]\n\n"
        f"Our current estimate for resolving this is [timeframe]. We will notify you immediately if anything changes.\n\n"
        f"If you have any further questions or need additional assistance, please feel free to reply directly to this email. "
        f"Our support team is always happy to help.\n\n"
        f"Thank you for your patience and for choosing [Company Name].\n\n"
        f"Kind regards,\n"
        f"Customer Support Team\n"
        f"[Company Name]\n\n"
        f"Use the available tools to investigate and resolve this issue. After tool execution, generate the professional customer email."
    )
    
    # Use LLM with tools directly (compatible approach)
    tools = [check_order_status, track_order, check_stock, initialize_resend, initialize_refund]
    llm_with_tools = llm.bind_tools(tools)
    
    # Maintaining rolling messages for tool-loop
    messages = [HumanMessage(content=task)]
    tool_messages = []
    detailed_reasoning = []
    result_text = ""
    while True:
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            break  # No more tool calls, exit loop

        print(f"\nResponse(please have tool_calls): {response}\n")
        print(f"\nTools available: {response.tool_calls if hasattr(response, 'tool_calls') else 'No tool calls'}\n")
        # Process tool calls if any
        #tool_messages = []
        #detailed_reasoning = []
        #result_text = ""

        if hasattr(response, 'tool_calls') and response.tool_calls:
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=response.tool_calls
                )
            )
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_input = tool_call['args']
                
                # Interruption of Critical Tools
                if tool_name in ['initialize_refund', 'initialize_resend']:
                    # Create the interrupt request
                    request = {
                        "action_request": {
                            "action": tool_call["name"],
                            "args": tool_call["args"]
                        },
                        "config": {
                                    "allow_ignore": True,
                                    "allow_respond": False,
                                    "allow_edit": False,
                                    "allow_accept": True
                                },
                        "description": f"""
                    Action Requires Approval

                    The support agent is attempting to perform a critical and high-impact order resolution action that requires your explicit approval.

                    Action Type: {tool_name}
                    """
                    }
                    print('\nEntered into the "FINAL BOSS INTERRUPT IF!! LESSSSGOOOOOO!!!"\n')
                    resp = interrupt(request)[0]

                    if resp["type"] == "accept":
                        pass # Do nothing, proceed to tool call, let it get executed
                    elif resp["type"] == "ignore":
                        # Log the denial but don't add to messages (prevents email extraction)
                        print(
                            f"Action {tool_name} denied by human reviewer for order {tool_input.get('order_id')}. Stopping resolution."
                        )
                        
                        # Return with denial state WITHOUT adding denial message
                        return {
                            "messages": [*state.messages, *tool_messages],  # Don't include denial_message
                            "action_taken": "Action Denied",
                            "reason": f"Human denied {tool_name} action",
                            "email_reply": None,  # ⬅️ Explicitly set to None
                            "requires_human_review": True,
                            "reasoning": {
                                **state.reasoning,
                                "resolve": f"{tool_name} denied by human - requires supervisor review"
                            },
                            "thought_process": state.thought_process + [{
                                "step": "resolve_issue",
                                "reasoning": f"Critical action {tool_name} was denied by human",
                                "output": "Execution stopped - requires supervisor intervention"
                            }]
                        }
                
                # Execute the tool
                tool_func = None
                for t in tools:
                    if t.__name__ == tool_name:
                        tool_func = t
                        break
                
                if tool_func:
                    # Call the tool
                    if isinstance(tool_input, dict):
                        tool_result = tool_func(**tool_input)
                    else:
                        tool_result = tool_func(tool_input)
                    
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

                    messages.append(
                        ToolMessage(
                                name=tool_name,
                                content=str(tool_result),
                                tool_call_id=tool_call["id"]
                            )
                    )
    # Get final response from LLM
    if response.content:
        result_text = response.content
    else:
        # Ask LLM to summarize based on tool results
        summary_prompt = RESOLUTION_SUMMARY_PROMPT.format(
            task=task,
            tool_results=detailed_reasoning
        )
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
        "email_reply": result_text,
        "reasoning": {**state.reasoning, "resolve": reasoning_summary},
        "thought_process": state.thought_process + [{
            "step": "resolve_issue",
            "reasoning": reasoning_summary,
            "detailed_steps": formatted_reasoning,
            "output": f"{action} - {reason}"
        }]
    }
