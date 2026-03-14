"""
Reward functions for Reinforcement Learning updates.
"""

def simple_gsm8k_reward(generated_text: str, target_text: str) -> float:
    """
    A simple reward function for demonstration purposes.
    In a precise scenario, this would extract the final numerical answer from the 
    generated text and compare it exactly to the target numerical answer, or use an LLM-as-a-judge.
    """
    # Very basic heuristic: check if numbers from target are in the generated text
    target_numbers = [word for word in target_text.split() if word.isdigit()]
    
    if not target_numbers:
        # If target has no numbers (unlikely for GSM8K but possible fallback), check string matching
        if any(char.isdigit() for char in generated_text):
           return 1.0
        return -0.5
        
    for num in target_numbers:
        if num in generated_text:
            return 1.0  # Positive reinforcement if target number is found
            
    return -1.0 # Strong penalty if the correct number isn't found
