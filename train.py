import torch
import config
from dataset import get_ttt_dataset
from rewards import simple_gsm8k_reward
from model import ParasiteMLPWrapper
from trl import PPOConfig, PPOTrainer

def train_test_time(ppo_model, tokenizer):
    """
    Production-grade Online Test-Time Training (TTT) using trl.PPOTrainer.
    The model receives a prompt, generates an output, calculates online reward, 
    and backpropagates via Proximal Policy Optimization (PPO).
    """
    dataset = get_ttt_dataset()
    
    ppo_config = PPOConfig(
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        mini_batch_size=config.MINI_BATCH_SIZE,
        init_kl_coef=config.KL_BETA,
        target_kl=0.1,
    )
    
    print("\nInitializing trl.PPOTrainer...")
    # Because we don't pass ref_model, PPOTrainer deepcopies ppo_model into a ref_model structure.
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        ref_model=None,
        tokenizer=tokenizer,
    )
    
    # CRITICAL INFERENTIAL ALIGNMENT:
    # We want the reference model to accurately represent the purely FROZEN BASE MODEL.
    # The deepcopy copied the parasite models as well.
    # By forcibly setting epsilon=0.0 on the ref_model, the parasite outputs are nullified,
    # meaning the ref_model behaves EXACTLY like the original, un-modified base Gemma 3 model!
    if ppo_trainer.ref_model is not None:
        for name, module in ppo_trainer.ref_model.named_modules():
            if isinstance(module, ParasiteMLPWrapper):
                module.epsilon = 0.0
                
    print(f"\nStarting Production Test-Time Training (PPO) for {config.TTT_STEPS} steps...")
    
    generation_kwargs = {
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.7, # Required for sampling logical trajectories in PPO
    }
    
    for i in range(config.TTT_STEPS):
        example = dataset[i]
        prompt = example['question']
        target = example['answer']
        
        # Format for Gemma Instruction format
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # PPO requires list of 1D tensors
        query_tensor = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][0].to(ppo_trainer.accelerator.device)
        
        # 1. Generate Response using the active policy
        print(f"Step {i+1} | Generating Response...")
        response_tensors = ppo_trainer.generate([query_tensor], return_prompt=False, **generation_kwargs)
        
        # 2. Decode response and calculate reward
        response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
        reward_val = simple_gsm8k_reward(response_text, target)
        reward_tensor = torch.tensor(reward_val, dtype=torch.float).to(ppo_trainer.accelerator.device)
        
        # 3. PPO Step (Calculates KL internally using ref_model, updates policy and value head)
        print(f"Step {i+1} | Executing PPO Step (Value Head, Policy Clip, KL Guardrails)...")
        stats = ppo_trainer.step([query_tensor], response_tensors, [reward_tensor])
        
        mean_reward = stats['env/reward_mean']
        kl_penalty = stats['objective/kl']
        ppo_loss = stats.get('ppo/loss/total', 0.0) # safely fallback if not tracked identically
        
        print(f"Step {i+1}/{config.TTT_STEPS} | TTT Reward: {mean_reward:5.2f} | KL: {kl_penalty:.4f} | PPO Loss: {ppo_loss:.4f}")
