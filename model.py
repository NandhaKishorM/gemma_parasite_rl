import torch
import torch.nn as nn
from unsloth import FastLanguageModel
import config

# =============================================================================
# 1. PARASITE POLICY NETWORK
# =============================================================================

class ParasitePolicyNetwork(nn.Module):
    """
    A lightweight memory/policy controller network.
    It takes the FFN activations (or input) and outputs a gating vector.
    """
    def __init__(self, hidden_size=config.HIDDEN_SIZE, bottleneck_size=config.BOTTLENECK_SIZE):
        super().__init__()
        # Small bottleneck network to keep parameter count low
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Linear(bottleneck_size, hidden_size)
        )
        
        # Initialize weights to zero so that initially, gating acts as an identity (no change)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class ParasiteMLPWrapper(nn.Module):
    """
    Wraps the Unsloth/Gemma MLP to inject the policy gating.
    h = original_mlp(x) * (1 + epsilon * tanh(policy(x)))
    """
    def __init__(self, original_mlp, hidden_size=config.HIDDEN_SIZE, epsilon=config.EPSILON):
        super().__init__()
        self.original_mlp = original_mlp
        self.policy = ParasitePolicyNetwork(hidden_size=hidden_size)
        self.epsilon = epsilon

    def forward(self, *args, **kwargs):
        # x is typically the first argument
        x = args[0]
        
        # 1. Pass through frozen base MLP
        with torch.no_grad():
            base_output = self.original_mlp(*args, **kwargs)
            
        # 2. Calculate policy gating based on the input features
        # Ensure we track gradients for the policy
        policy_out = self.policy(x)
        
        # 3. Multiplicative gating with strict magnitude limit (epsilon)
        # Bounding the gating to prevent complete memory loss / unintelligent strings
        g_l = 1.0 + self.epsilon * torch.tanh(policy_out)
        
        # 4. Modulate output
        return base_output * g_l

# =============================================================================
# 2. ARCHITECTURE SETUP
# =============================================================================

def setup_model():
    """
    Loads Gemma 3 1B using Unsloth, applies the parasite layers.
    """
    print(f"Loading Base {config.MODEL_NAME}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.MODEL_NAME,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=config.LOAD_IN_4BIT,
        )
    except Exception as e:
        print(f"Error loading model from Unsloth. Please check your token/setup: {e}")
        return None, None, []
        
    # Freeze the entire base model
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"Attaching Parasitic Policy Networks to layers: {config.TARGET_LAYERS}")
    
    parasite_params = []
    # Identify the MLP layers in Gemma 3 Unsloth Implementation
    for layer_idx in config.TARGET_LAYERS:
        original_mlp = model.model.layers[layer_idx].mlp
        
        wrapper = ParasiteMLPWrapper(original_mlp, hidden_size=config.HIDDEN_SIZE, epsilon=config.EPSILON)
        
        # Replace the original MLP with our wrapper
        model.model.layers[layer_idx].mlp = wrapper
        
        # Collect parameters for the optimizer
        parasite_params.extend(list(wrapper.policy.parameters()))
        
    total_params = sum(p.numel() for p in parasite_params)
    print(f"Parasite injected successfully. Trainable params: {total_params:,}")
    return model, tokenizer, parasite_params
