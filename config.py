"""
Configuration and hyperparameters for the Parasite Policy Test-Time Training.
"""

# Model Configuration
MODEL_NAME = "unsloth/gemma-3-1b-it"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = False  # Set to True for lower VRAM usage, False for bf16 speed

# Parasite Policy Configuration
HIDDEN_SIZE = 1152
BOTTLENECK_SIZE = 128
EPSILON = 0.04
TARGET_LAYERS = [4, 8, 12, 16, 20, 24] # Layers to attach the parasite policy

# Training Configuration
LEARNING_RATE = 5e-4
MAX_GRAD_NORM = 0.5
KL_BETA = 0.2  # Weight of the KL penalty to prevent catastrophic forgetting
TTT_STEPS = 5  # Number of online test-time training steps for demonstration
MAX_NEW_TOKENS = 40
BATCH_SIZE = 1
MINI_BATCH_SIZE = 1
PPO_EPOCHS = 1
