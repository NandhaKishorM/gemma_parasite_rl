"""
Reward functions for Reinforcement Learning updates.
"""

def simple_gsm8k_reward(generated_text: str, target_text: str) -> float:
    """
    A heuristic reward function for mathematical reasoning tasks like GSM8K.
    It extracts the final numerical answer from the generated text and compares
    it to the target numerical answer.
    """
    import re
    
    # GSM8K target text often looks like "#### 12". Extract it.
    target_match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', target_text)
    if not target_match:
        # Fallback if no #### formatting: just find the last number in target
        target_numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', target_text)
        if not target_numbers:
             return -0.5
        target_answer = target_numbers[-1].replace(',', '')
    else:
        target_answer = target_match.group(1).replace(',', '')

    # We prompted for "The answer is [number]"
    # Try to extract the number specifically from that statement
    gen_match = re.search(r'(?i)answer is\s*[:*]*\s*([$]*\s*-?\d+(?:,\d+)*(?:\.\d+)?)', generated_text)
    
    if gen_match:
        gen_answer = gen_match.group(1).replace('$', '').replace(',', '').strip()
        if gen_answer == target_answer:
            return 1.0
        else:
            return -1.0
    
    # Fallback: if the model failed to follow the final answer format, but the 
    # exact target answer exists SOMEWHERE in its generation, grant a partial reward.
    if target_answer in generated_text.replace(',', ''):
        return 0.5  
        
    return -1.0 # Strong penalty for incorrect math and incorrect formatting
