"""
Gemma 3 1B — Parasitic Policy Injection
Pure RL training: the parasite learns to steer the frozen model's behavior.
"""

import sys
import torch
import config
from model import setup_model
from train import train_rule_adherence, evaluate_adversarial

def main():
    print("=== Gemma 3 1B — Parasitic Policy Injection ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU. In Google Colab, select Runtime -> T4 GPU.")
    print(f"Target Device: {device.upper()}")

    # 1. Setup Model (base frozen + parasite injected)
    model, tokenizer, p_params = setup_model()

    if model is None:
        print("Failed to initialize model. Exiting.")
        sys.exit(1)

    model = model.to(device)

    # 2. Train the parasite with REINFORCE
    try:
        train_rule_adherence(model, tokenizer, p_params)
        evaluate_adversarial(model, tokenizer)
        print("\nParasitic Policy training complete.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
