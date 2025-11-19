from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from state import SupportAgentState
from tools import check_order_status, track_order, check_stock, initialize_resend, initialize_refund
from langchain_core.tools import tool as create_tool
import json
from typing import Dict, Any, List, Optional
from database.policies import format_policies_for_llm, get_policies_for_problem
from dotenv import load_dotenv
load_dotenv()
from database.memory import get_policies_context, get_products_context
from langgraph.types import interrupt

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



def tier_classifier(state: SupportAgentState):
    issue_text = state.messages[0].content
    
    prompt = (
        """# Customer Support Tier Classification System Prompt

You are an expert customer support tier classification AI. Your role is to analyze incoming customer issues and classify them into one of three support tiers based on complexity, required expertise, and potential business impact.

## Classification Criteria

### L1 (Basic/Tier 1) - Frontline Support
**Characteristics:**
- Simple, routine inquiries that can be resolved with standard procedures
- Issues covered by documented policies and FAQs
- No technical expertise required
- Can be resolved in a single interaction
- Low business impact

**Examples:**
- Order status checks
- Basic product information requests
- Standard returns/exchanges within policy
- Tracking information requests
- Password resets
- Shipping address updates
- General product availability questions
- Simple account inquiries

### L2 (Intermediate/Tier 2) - Specialized Support
**Characteristics:**
Handels more complex issues requiring deeper knowledge of products/services
- May involve troubleshooting, investigations, or policy exceptions
-When Tools like Refund or resend are called classifiy as L3

### L3 (Advanced/Tier 3) - Expert/Management Support
**Characteristics:**
- Highly complex or sensitive issues
- Requires senior judgment or policy exceptions
- Potential legal, fraud, Financial or security implications
- financial impact on business
**Examples:**
- Refund Requests
- Resend Requests
- Suspected fraud or chargebacks
- Legal threats or complaints
- Data privacy/security concerns (GDPR, data deletion)
- High-value order disputes (>$500)
- Repeated failures in service
- Product safety concerns
- Media/public relations issues
- refund disputes
- Account takeover or security breaches

## Classification Process

1. **Read the entire customer message carefully**
2. **Identify key indicators:**
   - What is the customer asking for?
   - What is the underlying problem?
   - What resolution are they seeking?
   - Is there urgency or emotional intensity?
   - Are there legal/security keywords?
   - What is the potential business impact?

3. **Apply decision logic:**
   - Start by assuming L1 unless evidence suggests otherwise
   - Escalate to L2 if: investigation needed, policy exceptions, coordination required
   - Escalate to L3 if: Refund or Resend tools are called

4. **Output format:**
   Respond with ONLY the tier level in your response: "L1", "L2", or "L3"
   Include brief reasoning (1-2 sentences) explaining your classification.

## Special Considerations

**Always escalate to L3 if:**
- Refund or Resend tools are called
- Customer mentions legal action, lawyers, or lawsuits
- Suspected fraud, account compromise, or security breach
- Data privacy requests (deletion, export under GDPR/CCPA)
- Product safety or health concerns
- Media involvement or public complaints
- Explicit VIP/premium customer status mentioned
- Order value exceeds $500
- Issue has been escalated multiple times before

**Default to L2 if uncertain** between L1 and L2 to ensure proper handling.

**Be conservative with L1** - only assign if you're confident it can be resolved with standard procedures.

## Example Classifications

**Example 1:**
Customer: "Where is my order #ORD12345? It was supposed to arrive yesterday."
Classification: **L1**
Reasoning: Standard order tracking inquiry, can be resolved by checking order status.

**Example 2:**
Customer: "I received a damaged Smart Watch (order #ORD67890). I need a replacement ASAP!"
Classification: **L3**
Reasoning: Requires stock verification and coordination for replacement/refund decision.

**Example 3:**
Customer: "This is the third time you've messed up my order! I'm contacting my lawyer and posting about this on social media. Order #ORD55555."
Classification: **L3**
Reasoning: Legal threat, repeated service failure, potential PR impact, requires senior management attention.

## Your Task

Analyze the provided customer issue and respond with:
1. The tier classification (L1, L2, or L3)
2. Brief reasoning for your decision (2-3 sentences maximum)

Be decisive, consistent, and err on the side of appropriate escalation to ensure customer satisfaction."""
    )
    
    response = llm.invoke([HumanMessage(content=f"{prompt}\nCustomer issue: {issue_text}")])
    response_text = response.content.strip().lower()

    if "l1" in response_text:
        tier_level = "L1"
    elif "l2" in response_text:
        tier_level = "L2"
    else:
        tier_level = "L3"

    # Interrupt and capture the decision
    if tier_level == "L3":
        decision = interrupt({
            "type": "tier_approval",
            "tier": tier_level,
            "message": f"This issue is classified as {tier_level}. Approve or Deny?",
            "options": ["Deny", "Approve"]
        })

    # Process the decision - handle various response formats
        approved = decision == "1"
        
        # Create message based on approval status
        if approved:
            status_msg = f"{tier_level} classification approved."
        else:
            status_msg = f"{tier_level} classification denied."
        
        return {
            "tier_level": tier_level,
            "approved": approved,
            "messages": [*state.messages, AIMessage(content=status_msg)]
        }
    else:
        return {
            "tier_level": tier_level,
            "approved": True,
            "messages": [*state.messages, AIMessage(content=f"{tier_level} classification automatically approved.")]
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
        "query_issue": answer,
        "is_support_ticket": True
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
    issue_text = state.messages[0].content
    problems_str = ", ".join(state.problems)
    classification_reasoning = state.reasoning.get("classify", "")

    # Fetch policies from memory store and inject as explicit context
    policies_context = get_policies_context()

    prompt = (
        "You are a support AI. Use the provided policy memory context to select the most appropriate policy.\n"
        "Do NOT assume hidden memory—only use what is shown.\n"
        "Return a clear choice and reasoning.\n\n"
        f"Customer Issue: {issue_text}\n"
        f"Problem Types: {problems_str}\n"
        f"Issue Analysis: {classification_reasoning}\n\n"
        "Policy Memory Context (from store):\n"
        f"{policies_context}"
    )

    structured_llm = llm.with_structured_output(PolicySelection)
    response = structured_llm.invoke([HumanMessage(content=prompt)])

    policy_name = response.policy_name
    policy_desc = response.policy_description
    reasoning = response.reasoning
    application_notes = response.application_notes or ""

    reasoning_message = AIMessage(content=f"🔍 **Policy Analysis**:\n{reasoning}")
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
    
    # Fetch product context snapshot from store for richer reasoning
    products_context = get_products_context()

    # Create task description
    task = (
        f"You are a customer support agent handling the following issue:\n"
        f"Customer issue: {issue_text}\n"
        f"Identified problem types: {problems_str}\n"
        f"Company policy: {policy_info}\n\n"
        f"Product Memory Context (from store):\n{products_context}\n\n"
        f"Instructions:\n"
        f"1. Extract order ID (format: ORD#####).\n"
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
        f"Resolution Summary:\n"
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
                        continue  # Skip this tool call
                
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
