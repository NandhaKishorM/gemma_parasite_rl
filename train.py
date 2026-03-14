import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Any
import config
from dataset import get_ttt_dataset
from rewards import simple_gsm8k_reward
from model import ParasiteMLPWrapper # Needed for instance checking

def compute_kl_penalty(logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes KL Divergence to constrain the policy from straying too far
    from the frozen base model's knowledge, preventing catastrophic forgetting.
    """
    # Use float32 for KL to prevent underflow
    p = F.softmax(base_logits.float(), dim=-1)
    log_q = F.log_softmax(logits.float(), dim=-1)
    # KL(P || Q) = sum(P * log(P/Q))
    kl = F.kl_div(log_q, p, reduction='batchmean')
    return kl

def train_test_time(model: torch.nn.Module, tokenizer: Any, parasite_params: List[torch.nn.Parameter]):
    """
    Online Test-Time Training (Inference learning).
    The model receives a prompt, generates an output, calculates online reward, 
    and backpropagates only to the Parasite Policy.
    """
    optimizer = AdamW(parasite_params, lr=config.LEARNING_RATE)
    dataset = get_ttt_dataset()
    
    model.train() 
    
    print(f"\nStarting Test-Time Training Loop for {config.TTT_STEPS} steps...")
    
    for i in range(config.TTT_STEPS):
        example = dataset[i]
        prompt = example['question']
        target = example['answer']
        
        # Format for Gemma Instruction format
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # =====================================================
        # 1. Base Model Pass (No policy interference for KL target)
        # =====================================================
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, ParasiteMLPWrapper):
                    module.epsilon = 0.0 # Temporarily Disable
                    
            base_outputs = model(**inputs)
            base_logits = base_outputs.logits
            
            for name, module in model.named_modules():
                if isinstance(module, ParasiteMLPWrapper):
                    module.epsilon = config.EPSILON # Re-enable
        
        # =====================================================
        # 2. Policy-controlled pass
        # =====================================================
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Simulate generating text to calculate reward (in real scenario, we sample trajectories)
        with torch.no_grad():
            gen_tokens = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            
        reward = simple_gsm8k_reward(generated_text, target)
        
        # =====================================================
        # 3. Guardrails & Loss Calculation
        # =====================================================
        # KL Guardrail (Crucial to prevent gibberish and preserve intelligence)
        kl_loss = compute_kl_penalty(logits, base_logits)
        
        # Objective Demo: Maximize Reward, Maximize similarity to base logits (KL beta)
        # Note: A true PPO implementation calculates Probability Ratios and log probs.
        # This proxy loss simulates the online direction.
        proxy_loss = (-1.0 * reward) + (config.KL_BETA * kl_loss)
        
        # Update Parasite Policies safely
        optimizer.zero_grad()
        
        # Perform backward on the generalized graph proxy
        (logits.mean() * proxy_loss.item()).backward()
        
        # Gradient clipping for absolute stability
        torch.nn.utils.clip_grad_norm_(parasite_params, max_norm=config.MAX_GRAD_NORM)
        optimizer.step()
        
        print(f"Step {i+1}/{config.TTT_STEPS} | TTT Reward: {reward:5.2f} | KL Penalty: {kl_loss.item():.4f} | Total Loss Proxy: {proxy_loss.item():.4f}")
