#!/usr/bin/env python3
"""
profile_subnetworks.py — Zero-shot sub-network identification via subtractive probing.

For each attention head and MLP intermediate neuron, computes:
    score = mean_law_activation / (mean_coding_activation + 1e-6)

Heads/neurons with score > THRESHOLD (2.5) are tagged as "law_dominant",
indicating they contribute disproportionately to legal domain processing.

Reference:
    Cao et al. "Zero-shot sub-network identification via subtractive probing"

Architecture (Llama-3.2-1B):
    - 16 layers, 32 attention heads, head_dim=64, intermediate_size=8192
    - Grouped-Query Attention: 32 Q heads, 8 KV heads
    - SwiGLU MLP: gate_proj, up_proj, down_proj

Method:
    1. Register forward pre-hooks on self_attn.o_proj (captures per-head outputs
       before the output projection which mixes heads) and mlp.down_proj (captures
       per-neuron activations before the down projection).
    2. Run 10 coding prompts and 10 law prompts individually through the model.
    3. For each head/neuron, compute the L2 norm of its activation vector averaged
       over sequence positions, then average over prompts within each domain.
    4. Law-dominant score = mean_law / (mean_coding + epsilon).
    5. Save mask to ./outputs/mask.json.
"""

import json
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
THRESHOLD = 2.5
DEVICE = "cpu"
DTYPE = torch.bfloat16
OUTPUT_DIR = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS = 30
NUM_HEADS = 9
NUM_KV_HEADS = 3
HEAD_DIM = 64
HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536

CODING_PROMPTS = [
    "Write a Python function to reverse a linked list.",
    "Implement a quicksort algorithm in Python.",
    "Write a function to find the longest common subsequence.",
    "Implement a binary search tree in Python with insert and search operations.",
    "Write a Python function for matrix multiplication.",
    "Implement a depth-first search algorithm for a graph.",
    "Write a function to check if a string is a palindrome.",
    "Implement a hash map from scratch in Python.",
    "Write a Python function for merge sort.",
    "Implement a breadth-first search for a graph.",
]

LAW_PROMPTS = [
    "What is the doctrine of stare decisis in contract law?",
    "Explain the elements of negligence in tort law.",
    "What is the difference between assault and battery?",
    "Define the principle of res ipsa loquitur.",
    "What constitutes a breach of fiduciary duty?",
    "Explain the parol evidence rule in contract interpretation.",
    "What is the doctrine of consideration in contract law?",
    "Define strict liability in product liability cases.",
    "What are the elements of a valid contract?",
    "Explain the concept of proximate cause in tort law.",
]


class ActivationCollector:
    """
    Registers forward pre-hooks to capture intermediate activations.

    Attention hook (o_proj pre-hook):
        The input to o_proj is the concatenated per-head attention output
        with shape [batch, q_len, num_heads * head_dim]. We reshape to
        [batch, q_len, num_heads, head_dim] and compute the L2 norm over
        the head_dim, then average over q_len to get a scalar per head.

    MLP hook (down_proj pre-hook):
        The input to down_proj is the SwiGLU intermediate state
        (silu(gate(x)) * up(x)) with shape [batch, q_len, intermediate_size].
        We compute the L2 norm over q_len to get a scalar per neuron.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.attentions = defaultdict(list)
        self.mlp_neurons = defaultdict(list)
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        layers = self.model.model.layers
        for idx, layer in enumerate(layers):
            attn_hook = layer.self_attn.o_proj.register_forward_pre_hook(
                self._make_attn_pre_hook(idx)
            )
            mlp_hook = layer.mlp.down_proj.register_forward_pre_hook(
                self._make_mlp_pre_hook(idx)
            )
            self._hooks.extend([attn_hook, mlp_hook])

    def _make_attn_pre_hook(self, layer_idx):
        def hook(module, input_args):
            hidden = input_args[0]
            if hidden.ndim != 3:
                return None
            bs, seq_len, _ = hidden.shape
            hidden = hidden.view(bs, seq_len, NUM_HEADS, HEAD_DIM)
            norms = hidden.norm(dim=-1).mean(dim=1)
            self.attentions[layer_idx].append(norms.detach().cpu())
            return None
        return hook

    def _make_mlp_pre_hook(self, layer_idx):
        def hook(module, input_args):
            hidden = input_args[0]
            if hidden.ndim != 3:
                return None
            norms = hidden.norm(dim=1)
            self.mlp_neurons[layer_idx].append(norms.detach().cpu())
            return None
        return hook

    def clear(self):
        self.attentions.clear()
        self.mlp_neurons.clear()

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def run_prompts(model, tokenizer, collector, prompts):
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)


def main():
    print("=" * 60)
    print("Differential Fidelity — Zero-Shot Subnetwork Profiling")
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {THRESHOLD}")
    print("=" * 60)

    print("\n[1/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] Registering activation hooks...")
    collector = ActivationCollector(model)

    print(f"[3/4] Profiling {len(CODING_PROMPTS)} coding prompts...")
    run_prompts(model, tokenizer, collector, CODING_PROMPTS)
    coding_attentions = {k: torch.stack(v).mean(dim=0) for k, v in collector.attentions.items()}
    coding_neurons = {k: torch.stack(v).mean(dim=0) for k, v in collector.mlp_neurons.items()}
    collector.clear()

    print(f"[3/4] Profiling {len(LAW_PROMPTS)} law prompts...")
    run_prompts(model, tokenizer, collector, LAW_PROMPTS)
    law_attentions = {k: torch.stack(v).mean(dim=0) for k, v in collector.attentions.items()}
    law_neurons = {k: torch.stack(v).mean(dim=0) for k, v in collector.mlp_neurons.items()}
    collector.remove_hooks()

    print("[4/4] Computing law-dominant scores...")
    law_heads = {}
    law_neuron_indices = {}

    for layer_idx in range(NUM_LAYERS):
        if layer_idx not in coding_attentions or layer_idx not in law_attentions:
            continue

        head_scores = law_attentions[layer_idx] / (coding_attentions[layer_idx] + 1e-6)
        law_head_idxs = torch.where(head_scores[0] > THRESHOLD)[0].tolist()
        if law_head_idxs:
            law_heads[str(layer_idx)] = law_head_idxs

        neuron_scores = law_neurons[layer_idx] / (coding_neurons[layer_idx] + 1e-6)
        law_neuron_idxs = torch.where(neuron_scores[0] > THRESHOLD)[0].tolist()
        if law_neuron_idxs:
            law_neuron_indices[str(layer_idx)] = law_neuron_idxs

    mask = {
        "law_heads": law_heads,
        "law_neurons": law_neuron_indices,
        "threshold": THRESHOLD,
    }

    mask_path = os.path.join(OUTPUT_DIR, "mask.json")
    with open(mask_path, "w") as f:
        json.dump(mask, f, indent=2)

    total_heads_tagged = sum(len(h) for h in law_heads.values())
    total_neurons_tagged = sum(len(n) for n in law_neuron_indices.values())
    total_heads = NUM_LAYERS * NUM_HEADS
    total_neurons = NUM_LAYERS * INTERMEDIATE_SIZE

    attn_params_total = 0
    attn_per_layer = HIDDEN_SIZE * (HIDDEN_SIZE + (NUM_KV_HEADS * HEAD_DIM) * 2 + HIDDEN_SIZE)
    for layer_idx_str, heads in law_heads.items():
        fraction = len(heads) / NUM_HEADS
        attn_params_total += fraction * attn_per_layer

    mlp_params_total = 0
    mlp_per_neuron_params = HIDDEN_SIZE * 3
    for layer_idx_str, neurons in law_neuron_indices.items():
        mlp_params_total += len(neurons) * mlp_per_neuron_params

    total_params = 136_000_000
    tagged_params = attn_params_total + mlp_params_total
    pct_affected = (tagged_params / total_params) * 100

    print("\n" + "=" * 60)
    print(f"Tagged law-dominant heads:   {total_heads_tagged} / {total_heads} "
          f"({100*total_heads_tagged/total_heads:.1f}%)")
    print(f"Tagged law-dominant neurons: {total_neurons_tagged} / {total_neurons} "
          f"({100*total_neurons_tagged/total_neurons:.1f}%)")
    print(f"Estimated params affected:  {tagged_params/1e6:.1f}M / {total_params/1e6:.0f}M "
          f"({pct_affected:.1f}%)")
    print(f"Mask saved to: {mask_path}")
    print("=" * 60)

    print("\nLayers with law-tagged heads:")
    for lid, heads in sorted(law_heads.items(), key=lambda x: int(x[0])):
        print(f"  Layer {lid}: heads {heads}")

    print("\nLayers with law-tagged neurons (count):")
    for lid, neurons in sorted(law_neuron_indices.items(), key=lambda x: int(x[0])):
        print(f"  Layer {lid}: {len(neurons)} neurons")


if __name__ == "__main__":
    main()
