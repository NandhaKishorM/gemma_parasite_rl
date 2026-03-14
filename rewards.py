"""
Reward functions for Reinforcement Learning updates.
"""

import re

def evaluate_reward(generated_text: str, target_text: str, task_type: str = "math") -> float:
    """
    A generic reward function router for Test-Time Training.
    Supports multiple task types natively without requiring external ML models.
    """
    if task_type == "math":
        return _math_reward(generated_text, target_text)
    elif task_type == "boolean":
        return _boolean_reward(generated_text, target_text)
    elif task_type == "exact_match":
        return _exact_match_reward(generated_text, target_text)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

def _math_reward(generated_text: str, target_text: str) -> float:
    """Extracts final numerical answers and compares them."""
    target_match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', target_text)
    if not target_match:
        target_numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', target_text)
        if not target_numbers: return -0.5
        target_answer = target_numbers[-1].replace(',', '')
    else:
        target_answer = target_match.group(1).replace(',', '')

    gen_match = re.search(r'(?i)answer is\s*[:*]*\s*([$]*\s*-?\d+(?:,\d+)*(?:\.\d+)?)', generated_text)
    if gen_match:
        gen_answer = gen_match.group(1).replace('$', '').replace(',', '').strip()
        return 1.0 if gen_answer == target_answer else -1.0
    
    if target_answer in generated_text.replace(',', ''):
        return 0.5  
        
    return -1.0

def _boolean_reward(generated_text: str, target_text: str) -> float:
    """Evaluates Yes/No or True/False answers."""
    target_clean = target_text.strip().lower()
    if target_clean not in ["yes", "no", "true", "false"]:
        return 0.0 # Target isn't boolean
        
    gen_match = re.search(r'(?i)answer is\s*[:*]*\s*(yes|no|true|false)', generated_text)
    if gen_match:
        gen_answer = gen_match.group(1).lower()
        return 1.0 if gen_answer == target_clean else -1.0
        
    # Fallback search
    if target_clean in generated_text.lower():
        return 0.5
    return -1.0

def _exact_match_reward(generated_text: str, target_text: str) -> float:
    """Evaluates categorical or multiple choice (e.g. A, B, C, D)"""
    target_clean = target_text.strip().lower()
    
    gen_match = re.search(r'(?i)answer is\s*[:*]*\s*([a-zA-Z0-9_]+)', generated_text)
    if gen_match:
        gen_answer = gen_match.group(1).lower()
        return 1.0 if gen_answer == target_clean else -1.0
        
    # Strict fallback
    if target_clean in generated_text.lower():
        return 0.5
    return -1.0
