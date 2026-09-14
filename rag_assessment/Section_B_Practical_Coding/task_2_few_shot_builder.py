"""
Section B — Task 2: Few-Shot Complaint Classifier Prompt Builder
Requirements:
- Define list of at least four labelled examples covering 4 categories:
  'Late Delivery', 'Wrong Item', 'Missing Item', 'Poor Quality'.
- Function build_few_shot_prompt(complaint_text) appending examples + complaint + blank Output line.
- Function add_example(text, label) appending new example pair.
- Demonstrate adding a new example and testing with 2 complaint strings.
"""

import sys

# Set stdout encoding for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global list of few-shot example dictionaries
FEW_SHOT_EXAMPLES = [
    {
        "input": "My pizza arrived 50 minutes after the estimated delivery time and was cold.",
        "output": "Late Delivery"
    },
    {
        "input": "I ordered a Veg Paneer Wrap but received a Chicken Burger instead.",
        "output": "Wrong Item"
    },
    {
        "input": "The delivery bag was missing the garlic bread and coke that I paid for.",
        "output": "Missing Item"
    },
    {
        "input": "The soup spilled all over the box and the salad smelled completely stale.",
        "output": "Poor Quality"
    }
]


def add_example(text: str, label: str):
    """
    Appends a new labelled complaint example to the few-shot examples bank.
    """
    allowed_labels = {"Late Delivery", "Wrong Item", "Missing Item", "Poor Quality"}
    if label not in allowed_labels:
        raise ValueError(f"Invalid label '{label}'. Must be one of {allowed_labels}")
        
    FEW_SHOT_EXAMPLES.append({"input": text.strip(), "output": label.strip()})
    print(f"✅ Added new example to prompt bank: [{label}] -> '{text}'")


def build_few_shot_prompt(complaint_text: str) -> str:
    """
    Constructs a few-shot prompt with all registered examples followed by the target complaint.
    """
    prompt = (
        "System: You are an AI Classifier for a food delivery platform. "
        "Classify the incoming customer complaint into exactly one of these categories: "
        "'Late Delivery', 'Wrong Item', 'Missing Item', or 'Poor Quality'.\n\n"
        "--- EXAMPLES ---\n"
    )
    
    for idx, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        prompt += f"Example {idx}:\nInput: \"{ex['input']}\"\nOutput: {ex['output']}\n\n"
        
    prompt += f"--- TARGET COMPLAINT ---\nInput: \"{complaint_text.strip()}\"\nOutput:"
    return prompt


if __name__ == "__main__":
    print("=== TASK 2 DEMONSTRATION ===\n")
    
    # Test Complaint 1
    complaint_1 = "I waited 1.5 hours and when the driver finally came, the fries were completely soggy."
    print("--- Test Case 1: Initial Few-Shot Prompt ---")
    prompt_1 = build_few_shot_prompt(complaint_1)
    print(prompt_1)
    print("\n" + "=" * 60 + "\n")
    
    # Adding a 5th example using add_example()
    print("--- Dynamically Adding New Example ---")
    add_example(
        text="The courier delivered the package to the wrong apartment building.",
        label="Late Delivery"
    )
    print("\n" + "=" * 60 + "\n")
    
    # Test Complaint 2 showing updated prompt structure
    complaint_2 = "My order was supposed to include 3 tacos but there were only 2 inside."
    print("--- Test Case 2: Prompt after Adding New Example ---")
    prompt_2 = build_few_shot_prompt(complaint_2)
    print(prompt_2)
    print("\n" + "=" * 60 + "\n")
