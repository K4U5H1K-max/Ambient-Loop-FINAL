"""
Customer Support Policies for ERP System
"""

# Dictionary of policies with their descriptions and when to use them
SUPPORT_POLICIES = {
    "Standard Return Policy": {
        "description": "Customers can return any item within 30 days of purchase for a full refund if the item is in its original condition.",
        "when_to_use": "For general return requests within the 30-day window where the item is unused or in original condition.",
        "applicable_problems": ["return", "refund", "general"]
    },
    "Damaged Item Policy": {
        "description": "If a customer receives a damaged or defective item, they are eligible for an immediate replacement or full refund, including shipping costs.",
        "when_to_use": "When a customer reports receiving a damaged or defective product, regardless of when the purchase was made.",
        "applicable_problems": ["damaged", "quality"]
    },
    "Non-Delivery Resolution": {
        "description": "If a package is marked as delivered but the customer hasn't received it, we'll initiate an investigation and either resend the item or provide a refund within 5 business days.",
        "when_to_use": "When tracking shows delivered but customer reports non-receipt of package.",
        "applicable_problems": ["non-delivery"]
    },
    "Delayed Shipment Compensation": {
        "description": "For orders delayed beyond the estimated delivery date by more than 3 business days, customers are eligible for expedited shipping on their next order or a 10% discount.",
        "when_to_use": "When shipping takes significantly longer than the estimated delivery timeframe.",
        "applicable_problems": ["delayed"]
    },
    "Wrong Item Resolution": {
        "description": "If a customer receives the wrong item, they can keep the incorrect item and we'll ship the correct one immediately, or they can return the wrong item for a full refund.",
        "when_to_use": "When a customer receives an item different from what they ordered.", #Wrong item pickup and right item pickup
        "applicable_problems": ["wrong-item"]
    },
    "Size/Fit Adjustment": {
        "description": "Customers can exchange clothing or wearable items for a different size within 45 days, with free return shipping.",
        "when_to_use": "For clothing or wearable items with size or fit issues.",
        "applicable_problems": ["fit"]
    },
    "Out-of-Stock Compensation": {
        "description": "If an item is out of stock after an order is placed, customers can choose to wait with a 15% discount, select an alternative item, or receive a full refund.",
        "when_to_use": "When inventory issues prevent fulfillment of an order as placed.",
        "applicable_problems": ["non-delivery", "delayed"]
    },
    "Premium Customer Service": {
        "description": "Customers who have spent over $500 in the past year receive priority support, free expedited shipping on replacements, and additional 5% compensation on any issues.",
        "when_to_use": "For high-value customers with any type of issue. Check customer purchase history.",
        "applicable_problems": ["general", "damaged", "delayed", "non-delivery", "wrong-item", "quality", "fit"]
    },
    "First-Time Customer Courtesy": {
        "description": "First-time customers receive extra flexibility on return timeframes (extended to 45 days) and a one-time courtesy refund or replacement even if outside normal policy guidelines.",
        "when_to_use": "For first-time customers experiencing any issues with their order.",
        "applicable_problems": ["general", "damaged", "delayed", "non-delivery", "wrong-item", "quality", "fit"]
    },
    "Technical Support Policy": {
        "description": "For products requiring technical setup, customers can schedule a free 15-minute consultation with our technical support team.",
        "when_to_use": "When customers have issues setting up or using electronic or complex products.",
        "applicable_problems": ["quality", "general"]
    },
    "Account Security Protocol": {
        "description": "For account-related issues, we require verification of identity through multiple factors before making any changes or providing sensitive information.",
        "when_to_use": "When handling account access, password resets, or personal information updates.",
        "applicable_problems": ["account"]
    },
    "Website Functionality Issues": {
        "description": "For reported website issues, we provide immediate workarounds when possible and escalate to the technical team with a 24-hour response commitment.",
        "when_to_use": "When customers report problems with the website functionality.",
        "applicable_problems": ["website"]
    }
}

def get_all_policies():
    """Return the complete policy dictionary"""
    return SUPPORT_POLICIES

def get_policy(policy_name):
    """Get a specific policy by name"""
    return SUPPORT_POLICIES.get(policy_name, None)

def get_policies_for_problem(problem_type):
    """Get all policies applicable to a specific problem type"""
    applicable_policies = {}
    for name, policy in SUPPORT_POLICIES.items():
        if problem_type in policy["applicable_problems"]:
            applicable_policies[name] = policy
    return applicable_policies

def format_policies_for_llm():
    """Format all policies in a way that's easy for an LLM to process"""
    formatted = "# Customer Support Policies\n\n"
    
    for name, policy in SUPPORT_POLICIES.items():
        formatted += f"## {name}\n"
        formatted += f"Description: {policy['description']}\n"
        formatted += f"When to use: {policy['when_to_use']}\n"
        formatted += f"Applicable problems: {', '.join(policy['applicable_problems'])}\n\n"
    
    return formatted
