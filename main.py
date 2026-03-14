"""
Gemma 3 1B - Test-Time Training with Parasite Policy
Supports both Math Benchmarking and System Prompt Replacement modes.
"""

import sys
import torch
import config
from model import setup_model
from train import train_test_time, train_rule_adherence, evaluate_adversarial

def main():
    print("=== Gemma 3 1B Parasite Policy Test-Time Training ===")
    print(f"Training Mode: {config.TRAINING_MODE.upper()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU. In Google Colab, select Runtime -> T4 GPU.")

    print(f"Target Device: {device.upper()}")

    # 1. Setup Model
    model, tokenizer, p_params = setup_model()

    if model is None:
        print("Failed to initialize model. Exiting.")
        sys.exit(1)

    model = model.to(device)

    # 2. Run Training Loop based on mode
    try:
        if config.TRAINING_MODE == "rule_adherence":
            train_rule_adherence(model, tokenizer, p_params)
            # Run adversarial evaluation after training
            evaluate_adversarial(model, tokenizer)
        elif config.TRAINING_MODE == "math":
            train_test_time(model, tokenizer, p_params)
        else:
            print(f"Unknown TRAINING_MODE: {config.TRAINING_MODE}")
            sys.exit(1)

        print("\nTest-Time Training (TTT) pipeline execution complete.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == "__main__":
    main()
