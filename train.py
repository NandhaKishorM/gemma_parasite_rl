import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Any
import config
from dataset import get_ttt_dataset
from rewards import simple_gsm8k_reward
from model import ParasiteMLPWrapper

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
    and backpropagates only to the Parasite Policy via REINFORCE proxy.
    """
    optimizer = AdamW(parasite_params, lr=config.LEARNING_RATE)
    dataset = get_ttt_dataset()
    
    model.train() 
    
    print(f"\nStarting Test-Time Training Loop (Functional PyTorch RL) for {config.TTT_STEPS} steps...")
    
    for i in range(config.TTT_STEPS):
        example = dataset[i]
        prompt = example['question']
        target = example['answer']
        
        # Format for Gemma Instruction format with explicit instruction for the reward parser
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt}\nThink step by step and end with 'The answer is [number]'.<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # =====================================================
        # 1. Base Model Pass (No policy interference for KL target & Eval)
        # =====================================================
        print(f"Step {i+1} | Executing Base Model Pass (Might compile kernels on 1st run)...")
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, ParasiteMLPWrapper):
                    module.epsilon = 0.0 # Temporarily Disable
                    
            base_outputs = model(**inputs)
            base_logits = base_outputs.logits
            
            # Generate Base Model Response for benchmarking
            base_gen_tokens = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
            input_len = inputs['input_ids'].shape[1]
            base_generated_text = tokenizer.decode(base_gen_tokens[0][input_len:], skip_special_tokens=True)
            base_reward = simple_gsm8k_reward(base_generated_text, target)
            
            for name, module in model.named_modules():
                if isinstance(module, ParasiteMLPWrapper):
                    module.epsilon = config.EPSILON # Re-enable
        
        # =====================================================
        # 2. Policy-controlled pass
        # =====================================================
        print(f"Step {i+1} | Executing Policy-controlled Pass...")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Simulate generating text to calculate reward
        print(f"Step {i+1} | Generating Response ({config.MAX_NEW_TOKENS} max tokens)...")
        with torch.no_grad():
            gen_tokens = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
            # Slice off the input prompt to get pure generated sequence
            input_len = inputs['input_ids'].shape[1]
            generated_text = tokenizer.decode(gen_tokens[0][input_len:], skip_special_tokens=True)
            
        print(f"\n--- [EVALUATION] ---")
        print(f"Q: {prompt[:120]}...\nTarget: {target}")
        
        # Determine actual math answer extraction for neat printing
        import re
        def extract_answer(t):
             match = re.search(r'(?i)answer is\s*[:*]*\s*([$]*\s*-?\d+(?:,\d+)*(?:\.\d+)?)', t)
             return match.group(1).strip() if match else "(No valid final format)"
             
        base_ext = extract_answer(base_generated_text)
        pol_ext = extract_answer(generated_text)
        
        print(f"\n[FROZEN BASE GEN]: ... {base_generated_text[-120:]}")
        print(f"  > Extracted Final Answer: {base_ext} | Reward: {base_reward}")
        
        print(f"\n[PARASITE POLICY GEN]: ... {generated_text[-120:]}")
        print(f"  > Extracted Final Answer: {pol_ext} | Reward: {reward}")
        print(f"--------------------\n")
        
        # =====================================================
        # 3. Guardrails & Loss Calculation
        # =====================================================
        print(f"Step {i+1} | Calculating Guardrails & Updating Policy...")
        # KL Guardrail (Crucial to prevent gibberish and preserve intelligence)
        kl_loss = compute_kl_penalty(logits, base_logits)
        
        # Objective Demo: Maximize Reward, Maximize similarity to base logits (KL beta)
        proxy_loss = (-1.0 * reward) + (config.KL_BETA * kl_loss)
        
        # Update Parasite Policies safely
        optimizer.zero_grad()
        
        # Perform backward on the generalized graph proxy
        (logits.mean() * proxy_loss.item()).backward()
        
        # Gradient clipping for absolute stability
        torch.nn.utils.clip_grad_norm_(parasite_params, max_norm=config.MAX_GRAD_NORM)
        optimizer.step()
        
        print(f"Step {i+1}/{config.TTT_STEPS} | TTT Reward: {reward:5.2f} | KL Penalty: {kl_loss.item():.4f} | Total Loss Proxy: {proxy_loss.item():.4f}")
