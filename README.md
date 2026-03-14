# Gemma 3 1B - Test-Time Training (TTT) via Parasitic Policy

This repository implements an advanced reinforcement learning architecture designed to steer a frozen LLM (Gemma 3 1B) during inference. 

Based on research from late 2025 and early 2026, it utilizes a "Parasitic FFN Policy" that wraps the intermediate layers of the Transformer without altering the base weights. This enables dynamic behavior modification (via test-time training) while fiercely preventing catastrophic forgetting through rigorous KL divergence guardrails and strict multiplicative gating limits (`epsilon`).

## Setup Requirements

This project is tailored for **Google Colab** using an **NVIDIA T4 GPU**, leveraging [Unsloth](https://github.com/unslothai/unsloth) to radically reduce memory footprint and speed up execution.

### Installation

```bash
pip install -r requirements.txt
```

*(Note: Ensure you are running Python 3.10+ and have CUDA libraries correctly configured by the Colab environment).*

## Project Structure

*   `main.py`: The main entry point to initialize the model and start the test-time training loop.
*   `model.py`: Contains the architecture for the `ParasitePolicyNetwork` and the `ParasiteMLPWrapper` that intercepts the Unsloth/Gemma FFN activations.
*   `train.py`: Contains the core Reinforcement Learning Test-Time loop and `KL Divergence` calculation.
*   `dataset.py`: Handles fetching reasoning datasets (e.g., `openai/gsm8k`) for the online updates.
*   `rewards.py`: Contains custom heuristic and verifiable reward functions mapping text generation to training signals.
*   `config.py`: Exposes all dynamic hyper-parameters, making it easy to tune the bottleneck size, epsilon bounds, learning rates, or targeting layers.

## Usage

Run the training pipeline simply by executing:

```bash
python main.py
```

## How It Works

1.  **Frozen Substrate:** Most of the 1 billion parameters of the base Gemma 3 model are entirely frozen. Gradient calculation is disabled for these layers.
2.  **Parasite Injection:** We insert a tiny, trainable bottleneck network at layers `[4, 8, 12, 16, 20, 24]`.
3.  **Forward Pass Gating:** During inference, `h_modified = h_base * (1 + epsilon * tanh(policy(x)))`. The policy computes a slight steering vector based on the layer's internal activation state.
4.  **Online Update:** As the model answers a prompt, the reward is calculated online. The network back-propagates to update only the small parasite network weights. To avoid memory degradation (gibberish string outputs), a heavy KL divergence penalty restricts the policy from drastically altering the original output distribution.
