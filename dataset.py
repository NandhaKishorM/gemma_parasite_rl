"""
Dataset loading and processing for Test-Time Training.
"""
from datasets import load_dataset

def get_ttt_dataset(dataset_name="openai/gsm8k", config_name="main", split="train"):
    """
    Loads a reasoning dataset (like GSM8K) to train the policy during Test-Time.
    Mathematical reasoning scenarios are excellent for steering the policy to improve reasoning.
    """
    print(f"Loading test-time training dataset ({dataset_name})...")
    try:
        dataset = load_dataset(dataset_name, config_name, split=split)
        return dataset
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}. Error: {e}")
        print("Falling back to a dummy dataset for demonstration.")
        return [{'question': 'What is 2+2?', 'answer': '4'}] * 10
