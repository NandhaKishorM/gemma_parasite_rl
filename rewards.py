"""
Reward functions for Reinforcement Learning updates.
Supports math benchmarks, boolean/categorical tasks, and rule adherence (system prompt replacement).
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
    elif task_type == "rule_adherence":
        return _rule_adherence_reward(generated_text, target_text)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


# =========================================================================
# Rule Adherence Reward (System Prompt Replacement)
# =========================================================================
def _rule_adherence_reward(generated_text: str, rules_json: str) -> float:
    """
    Evaluates whether the model's response adheres to a set of business rules.
    The 'target_text' parameter here is a JSON string encoding the active rules.
    This replaces system prompts by baking behavioral rules into the Parasite Policy.
    """
    import json
    rules = json.loads(rules_json)
    
    gen_lower = generated_text.lower()
    total_score = 0.0
    num_rules = len(rules)
    
    if num_rules == 0:
        return 0.0

    for rule in rules:
        rule_score = 1.0  # Start with perfect score for this rule
        
        # Check prohibited keywords (critical violations = harsh penalty)
        prohibit = rule.get("prohibit_keywords", [])
        for keyword in prohibit:
            if keyword.lower() in gen_lower:
                rule_score = -1.0  # Instant failure for this rule
                break
        
        # Check enforced keywords (must be present)
        if rule_score > 0:
            enforce = rule.get("enforce_keywords", [])
            if enforce:
                found = sum(1 for kw in enforce if kw.lower() in gen_lower)
                if found == 0:
                    rule_score = -0.5  # Partial penalty: none of the required keywords found
                elif found < len(enforce):
                    rule_score = 0.5   # Partial credit: some but not all keywords found
                # else: rule_score stays 1.0 (all keywords found)
        
        total_score += rule_score
    
    # Normalize to [-1.0, 1.0] range
    normalized = total_score / num_rules
    return round(normalized, 4)


# =========================================================================
# Math Reward (GSM8K, AIME)
# =========================================================================
def _math_reward(generated_text: str, target_text: str) -> float:
    """Extracts final numerical answers and compares them."""
    target_match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', target_text)
    if not target_match:
        target_numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', target_text)
        if not target_numbers: return -0.5
        target_answer = target_numbers[-1].replace(',', '')
    else:
        target_answer = target_match.group(1).replace(',', '')

    gen_match = re.search(r'(?i)answer is[^0-9]*(-?\d+(?:,\d+)*(?:\.\d+)?)', generated_text)
    if gen_match:
        gen_answer = gen_match.group(1).replace(',', '').strip()
        return 1.0 if gen_answer == target_answer else -1.0
    
    if target_answer in generated_text.replace(',', ''):
        return 0.5  
        
    return -1.0


# =========================================================================
# Boolean Reward (TruthfulQA, StrategyQA)
# =========================================================================
def _boolean_reward(generated_text: str, target_text: str) -> float:
    """Evaluates Yes/No or True/False answers."""
    target_clean = target_text.strip().lower()
    if target_clean not in ["yes", "no", "true", "false"]:
        return 0.0

    gen_match = re.search(r'(?i)answer is\s*[:*]*\s*(yes|no|true|false)', generated_text)
    if gen_match:
        gen_answer = gen_match.group(1).lower()
        return 1.0 if gen_answer == target_clean else -1.0

    if target_clean in generated_text.lower():
        return 0.5
    return -1.0


# =========================================================================
# Exact Match Reward (MMLU, ARC)
# =========================================================================
def _exact_match_reward(generated_text: str, target_text: str) -> float:
    """Evaluates categorical or multiple choice (e.g. A, B, C, D)"""
    target_clean = target_text.strip().lower()

    gen_match = re.search(r'(?i)answer is\s*[:*]*\s*([a-zA-Z0-9_]+)', generated_text)
    if gen_match:
        gen_answer = gen_match.group(1).lower()
        return 1.0 if gen_answer == target_clean else -1.0

    if target_clean in generated_text.lower():
        return 0.5
    return -1.0
