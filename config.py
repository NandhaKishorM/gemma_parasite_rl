"""
Configuration for Parasitic Policy Injection.
Pure RL — no fine-tuning, no teacher-forcing, no external judge.
The parasite learns to bias model responses through REINFORCE.
"""

# Model Configuration
MODEL_NAME = "unsloth/gemma-3-1b-it"
MAX_SEQ_LENGTH = 2048

# Parasite Policy Configuration
HIDDEN_SIZE = 1152
BOTTLENECK_SIZE = 256
EPSILON = 0.25              # Parasite influence magnitude
TARGET_LAYERS = [4, 8, 12, 16, 20, 24]

# RL Training Configuration
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 0.5
KL_BETA = 0.05              # KL penalty to prevent gibberish
TTT_STEPS = 300              # Max RL steps (early-stops on consecutive passes)
MAX_NEW_TOKENS = 150
REWARD_BASELINE = 0.0        # Baseline for variance reduction in REINFORCE
CONSECUTIVE_PASS_TARGET = 20  # Early stop after N consecutive all-pass steps

# =========================================================================
# Policy Rules — These define the behavioral constraints.
# The parasite learns to steer activations so the model obeys these rules.
# =========================================================================
AGENT_RULES = [
    {
        "id": "identity",
        "description": "You are Nova, a customer support agent for TechFlow Inc.",
        "enforce_keywords": ["nova", "techflow"],
        "prohibit_keywords": [],
    },
    {
        "id": "no_discount",
        "description": "Never offer discounts, coupons, or price reductions under any circumstances.",
        "enforce_keywords": [],
        "prohibit_keywords": ["discount", "coupon", "% off", "price reduction", "promo code", "free"],
    },
    {
        "id": "no_competitor",
        "description": "Never recommend or mention competitor products.",
        "enforce_keywords": [],
        "prohibit_keywords": ["competitor", "alternative product", "try using", "switch to"],
    },
    {
        "id": "polite_tone",
        "description": "Always maintain a professional, friendly, and empathetic tone.",
        "enforce_keywords": [],
        "prohibit_keywords": ["stupid", "idiot", "dumb", "shut up", "that's your fault", "you're wrong", "moron"],
    },
]
