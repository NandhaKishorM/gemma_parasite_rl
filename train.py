import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Any
import json
import re
import config
from dataset import get_ttt_dataset
from rewards import evaluate_reward
from model import ParasiteMLPWrapper


def compute_kl_penalty(logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes KL Divergence to constrain the policy from straying too far
    from the frozen base model's knowledge, preventing catastrophic forgetting.
    """
    p = F.softmax(base_logits.float(), dim=-1)
    log_q = F.log_softmax(logits.float(), dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean')
    return kl


def _get_active_rules(target_rule_ids: list) -> list:
    """Fetches the full rule objects from config for the given rule IDs."""
    return [r for r in config.AGENT_RULES if r["id"] in target_rule_ids]


def _format_rules_description(rules: list) -> str:
    """Formats rules into a human-readable string for debug output."""
    return " | ".join([f"[{r['id']}]" for r in rules])


# =========================================================================
# Rule Adherence Training (System Prompt Replacement Mode)
# =========================================================================
def train_rule_adherence(model: torch.nn.Module, tokenizer: Any, parasite_params: List[torch.nn.Parameter]):
    """
    Test-Time Training loop for System Prompt Replacement.
    Instead of training on math datasets, we train on adversarial scenarios
    that attempt to trick the model into breaking business rules.
    The Parasite Policy learns to enforce these rules at the activation level.
    """
    optimizer = AdamW(parasite_params, lr=config.LEARNING_RATE)
    scenarios = config.ADVERSARIAL_SCENARIOS
    num_scenarios = len(scenarios)
    model.train()

    print(f"\n{'='*60}")
    print(f"  SYSTEM PROMPT REPLACEMENT — Rule Adherence Training")
    print(f"  Rules to encode: {len(config.AGENT_RULES)}")
    print(f"  Adversarial scenarios: {num_scenarios}")
    print(f"  Training steps: {config.TTT_STEPS}")
    print(f"{'='*60}\n")

    # Print rules being encoded
    for rule in config.AGENT_RULES:
        prohibit = ", ".join(rule["prohibit_keywords"]) if rule["prohibit_keywords"] else "(none)"
        enforce = ", ".join(rule["enforce_keywords"]) if rule["enforce_keywords"] else "(none)"
        print(f"  Rule [{rule['id']}]: {rule['description']}")
        print(f"    Enforce: {enforce} | Prohibit: {prohibit}")
    print()

    stats = {"total_reward": 0.0, "violations": 0, "steps": 0}

    for i in range(config.TTT_STEPS):
        scenario = scenarios[i % num_scenarios]
        user_prompt = scenario["prompt"]
        active_rules = _get_active_rules(scenario["target_rules"])
        rules_json = json.dumps(active_rules)

        # No system prompt! The model gets a raw user message with zero instructions.
        formatted_prompt = f"<bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # =====================================================
        # 1. Base Model Pass (epsilon=0 for KL target & comparison)
        # =====================================================
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, ParasiteMLPWrapper):
                    module.epsilon = 0.0

            base_outputs = model(**inputs)
            base_logits = base_outputs.logits

            base_gen_tokens = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
            input_len = inputs['input_ids'].shape[1]
            base_text = tokenizer.decode(base_gen_tokens[0][input_len:], skip_special_tokens=True)
            base_reward = evaluate_reward(base_text, rules_json, "rule_adherence")

            for name, module in model.named_modules():
                if isinstance(module, ParasiteMLPWrapper):
                    module.epsilon = config.EPSILON

        # =====================================================
        # 2. Policy-controlled pass
        # =====================================================
        outputs = model(**inputs)
        logits = outputs.logits

        with torch.no_grad():
            gen_tokens = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
            input_len = inputs['input_ids'].shape[1]
            policy_text = tokenizer.decode(gen_tokens[0][input_len:], skip_special_tokens=True)

        reward = evaluate_reward(policy_text, rules_json, "rule_adherence")

        # Track violations
        if reward < 0.5:
            stats["violations"] += 1
        stats["total_reward"] += reward
        stats["steps"] += 1

        # =====================================================
        # 3. Debug Output — Side-by-side comparison
        # =====================================================
        print(f"╔══ Step {i+1}/{config.TTT_STEPS} ═══════════════════════════════════")
        print(f"║ Jailbreak Attempt: \"{user_prompt}\"")
        print(f"║ Testing Rules: {_format_rules_description(active_rules)}")
        print(f"╠── FROZEN BASE (no rules) ──────────────────────────")
        print(f"║ {base_text[:200]}")
        print(f"║ >> Base Reward: {base_reward:+.2f}")
        print(f"╠── PARASITE POLICY (rules learned) ────────────────")
        print(f"║ {policy_text[:200]}")
        print(f"║ >> Policy Reward: {reward:+.2f}")

        # Show which specific rules were violated
        for rule in active_rules:
            policy_lower = policy_text.lower()
            violated_keywords = [kw for kw in rule.get("prohibit_keywords", []) if kw.lower() in policy_lower]
            missing_keywords = [kw for kw in rule.get("enforce_keywords", []) if kw.lower() not in policy_lower]
            status = "✅ PASS" if not violated_keywords and not missing_keywords else "❌ FAIL"
            detail = ""
            if violated_keywords:
                detail += f" Prohibited words used: {violated_keywords}"
            if missing_keywords:
                detail += f" Missing required words: {missing_keywords}"
            print(f"║   [{rule['id']}] {status}{detail}")

        print(f"╚═══════════════════════════════════════════════════\n")

        # =====================================================
        # 4. Guardrails & Policy Update
        # =====================================================
        kl_loss = compute_kl_penalty(logits, base_logits)
        proxy_loss = (-1.0 * reward) + (config.KL_BETA * kl_loss)

        optimizer.zero_grad()
        (logits.mean() * proxy_loss.item()).backward()
        torch.nn.utils.clip_grad_norm_(parasite_params, max_norm=config.MAX_GRAD_NORM)
        optimizer.step()

    # =====================================================
    # Final Summary
    # =====================================================
    avg_reward = stats["total_reward"] / max(stats["steps"], 1)
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Average Reward: {avg_reward:+.4f}")
    print(f"  Rule Violations: {stats['violations']}/{stats['steps']} steps")
    print(f"  The Parasite Policy has now internalized {len(config.AGENT_RULES)} rules.")
    print(f"  System prompt is NO LONGER NEEDED for inference.")
    print(f"{'='*60}\n")


# =========================================================================
# Math Benchmark Training (Original Mode)
# =========================================================================
def train_test_time(model: torch.nn.Module, tokenizer: Any, parasite_params: List[torch.nn.Parameter]):
    """
    Online Test-Time Training for math benchmarks (AIME, GSM8K).
    """
    optimizer = AdamW(parasite_params, lr=config.LEARNING_RATE)
    dataset = get_ttt_dataset()
    model.train()

    print(f"\nStarting Test-Time Training Loop (Math Mode) for {config.TTT_STEPS} steps...")

    for i in range(config.TTT_STEPS):
        example = dataset[i]
        prompt = example.get('question', example.get('problem', ''))
        target = str(example.get('answer', example.get('solution', '')))

        task_type = "math"
        hint = "\nThink step by step and end with 'The answer is [number]'."
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt}{hint}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 1. Base Model Pass
        print(f"Step {i+1} | Executing Base Model Pass...")
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, ParasiteMLPWrapper):
                    module.epsilon = 0.0

            base_outputs = model(**inputs)
            base_logits = base_outputs.logits

            base_gen_tokens = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
            input_len = inputs['input_ids'].shape[1]
            base_generated_text = tokenizer.decode(base_gen_tokens[0][input_len:], skip_special_tokens=True)
            base_reward = evaluate_reward(base_generated_text, target, task_type)

            for name, module in model.named_modules():
                if isinstance(module, ParasiteMLPWrapper):
                    module.epsilon = config.EPSILON

        # 2. Policy-controlled pass
        print(f"Step {i+1} | Executing Policy-controlled Pass...")
        outputs = model(**inputs)
        logits = outputs.logits

        print(f"Step {i+1} | Generating Response ({config.MAX_NEW_TOKENS} max tokens)...")
        with torch.no_grad():
            gen_tokens = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
            input_len = inputs['input_ids'].shape[1]
            generated_text = tokenizer.decode(gen_tokens[0][input_len:], skip_special_tokens=True)

        reward = evaluate_reward(generated_text, target, task_type)

        def extract_answer(t):
            match = re.search(r'(?i)answer is[^0-9]*(-?[0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)', t)
            return match.group(1).strip() if match else "(No final answer)"

        print(f"\n--- [EVALUATION] ---")
        print(f"Q: {prompt[:120]}...\nTarget: {target}")
        print(f"\n[FROZEN BASE]: ... {base_generated_text[-120:]}")
        print(f"  > Answer: {extract_answer(base_generated_text)} | Reward: {base_reward}")
        print(f"\n[PARASITE POLICY]: ... {generated_text[-120:]}")
        print(f"  > Answer: {extract_answer(generated_text)} | Reward: {reward}")
        print(f"--------------------\n")

        # 3. Guardrails & Loss
        kl_loss = compute_kl_penalty(logits, base_logits)
        proxy_loss = (-1.0 * reward) + (config.KL_BETA * kl_loss)

        optimizer.zero_grad()
        (logits.mean() * proxy_loss.item()).backward()
        torch.nn.utils.clip_grad_norm_(parasite_params, max_norm=config.MAX_GRAD_NORM)
        optimizer.step()

        print(f"Step {i+1}/{config.TTT_STEPS} | Reward: {reward:5.2f} | KL: {kl_loss.item():.4f} | Loss: {proxy_loss.item():.4f}")
