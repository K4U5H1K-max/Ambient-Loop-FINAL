# Centralized system prompts for the customer support agent

# Ticket validation/classification
CLASSIFICATION_PROMPT = """Analyze this customer message and extract:
1. Whether it's a support ticket (order issues, product problems, complaints, refunds, etc.)
2. Whether an order ID is present (format: ORD##### where # is a digit)
3. The exact order ID if present
Respond with structured JSON."""

# Tier classification prompt
TIER_CLASSIFIER_PROMPT = """# Customer Support Tier Classification System Prompt
... (full details to be extracted from nodes.py/backup.py) ..."""

# Policy selection prompt (structured)
POLICY_SELECTION_PROMPT = """You are a support AI. ... (full details to be added here as per source) ..."""

# Issue classification prompt
ISSUE_CLASSIFIER_PROMPT = """You are a customer support AI Agent. Analyze the following customer issue and identify problem types. Select from the following categories:
- non-delivery
- delayed
- damaged
- wrong-item
- quality
- fit
- return
- refund
- account
- website
- general
"""

# Resolution summary / customer email template
RESOLUTION_SUMMARY_PROMPT = """Based on the investigation, provide a resolution for the customer.

Task: {{task}}

Tool results: {{detailed_reasoning}}

Resolution Summary:
Dear [Customer Name],
Thank you for reaching out to us. We appreciate you contacting [Company Name]. ... (fill in as in nodes)
"""

# Add any additional prompts as found in nodes.py/backup.py.
