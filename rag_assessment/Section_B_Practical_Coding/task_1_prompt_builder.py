"""
Section B — Task 1: Structured Prompt Builder for Food Delivery Chatbot
Requirements:
- Define System Prompt assigning role of food delivery support agent, professional tone, max 80 words limit.
- Formatted user message template accepting `customer_name`, `order_id`, and `issue_type`.
- Populate template with at least two test cases and print full prompt pair for each.
- Validation function raising ValueError if issue_type NOT IN ['late delivery', 'missing item', 'wrong item'].
- Try-except block catching ValueError and printing user-friendly error message.
"""

import sys

# Set stdout encoding for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Allowed issue types for validation
ALLOWED_ISSUE_TYPES = {"late delivery", "missing item", "wrong item"}


def build_system_prompt() -> str:
    """
    Returns the role-based system prompt for food delivery support.
    """
    return (
        "System Prompt:\n"
        "You are an AI Customer Support Agent for QuickBite Food Delivery. "
        "Your role is to assist customers professionally, empathetically, and concisely. "
        "Follow all platform policy guidelines. Your responses MUST NOT exceed 80 words in length."
    )


def validate_issue_type(issue_type: str) -> str:
    """
    Validates that the issue type is one of the allowed values.
    Raises ValueError if invalid.
    """
    normalized = issue_type.strip().lower()
    if normalized not in ALLOWED_ISSUE_TYPES:
        allowed_str = ", ".join(f"'{i}'" for i in sorted(ALLOWED_ISSUE_TYPES))
        raise ValueError(
            f"Invalid issue_type '{issue_type}'. Issue type must be one of: {allowed_str}."
        )
    return normalized


def build_user_prompt(customer_name: str, order_id: str, issue_type: str) -> str:
    """
    Validates inputs and returns the formatted user message prompt.
    """
    valid_issue = validate_issue_type(issue_type)
    
    user_prompt = (
        f"User Message:\n"
        f"Customer Name: {customer_name}\n"
        f"Order ID: {order_id}\n"
        f"Reported Issue: {valid_issue.title()}\n"
        f"Request: Please review my order details and advise on the resolution process."
    )
    return user_prompt


def generate_support_prompt(customer_name: str, order_id: str, issue_type: str):
    """
    Constructs and prints the complete structured prompt pair inside a try-except block.
    """
    print("=" * 60)
    print(f"Generating Prompt for: {customer_name} | Order: {order_id} | Issue: '{issue_type}'")
    print("=" * 60)
    
    try:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(customer_name, order_id, issue_type)
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        print(full_prompt)
        print("-" * 60 + "\n")
        return full_prompt
        
    except ValueError as e:
        print(f"❌ VALIDATION ERROR: {e}")
        print("   --> Prompt construction aborted safely without crashing.")
        print("-" * 60 + "\n")
        return None


if __name__ == "__main__":
    print("=== TASK 1 DEMONSTRATION ===\n")
    
    # Test Case 1: Valid - Late Delivery
    generate_support_prompt(
        customer_name="Rahul Sharma",
        order_id="QB-98421",
        issue_type="late delivery"
    )
    
    # Test Case 2: Valid - Missing Item
    generate_support_prompt(
        customer_name="Priya Patel",
        order_id="QB-77104",
        issue_type="missing item"
    )
    
    # Test Case 3: Invalid - Issue type not in allowed set (triggers ValueError handling)
    generate_support_prompt(
        customer_name="Amit Kumar",
        order_id="QB-55412",
        issue_type="rude driver behavior"
    )
