"""
Gemma 3 1B - Test-Time Training with Parasite Policy
Designed to be executable as a standalone module.
"""

import sys
import torch
from model import setup_model
from train import train_test_time

def main():
    print("=== Gemma 3 1B Parasite Policy Test-Time Training ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU. In Google Colab, select Runtime -> T4 GPU.")
        
    print(f"Target Device: {device.upper()}")
    
    # 1. Setup Model
    ppo_model, tokenizer = setup_model()
    
    if ppo_model is None:
        print("Failed to initialize model. Exiting.")
        sys.exit(1)
        
    ppo_model = ppo_model.to(device)
    
    # 2. Run Test-Time Learning Loop
    try:
        train_test_time(ppo_model, tokenizer)
        print("\nTest-Time Training (TTT) pipeline execution complete.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == "__main__":
    main()
