#!/usr/bin/env python3
"""
profile_subnetworks.py — Zero-shot sub-network identification via Cohen's d effect size.

For each attention head and MLP intermediate neuron, computes:
    d = (mean_law - mean_coding) / sqrt((var_law + var_coding) / 2 + 1e-6)

Units with d > THRESHOLD are tagged as "law_dominant".
Also computes domain prototype embeddings (mean of last hidden state)
used by the engine's embedding-based router.

Reference:
    Cao et al. "Zero-shot sub-network identification via subtractive probing"
    Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
"""

import json
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
THRESHOLD = 1.0
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

TRAIN_CODING = [
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

TRAIN_LAW = [
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

HELD_OUT_CODING = [
    "Write a Python decorator that measures function execution time.",
    "Implement a trie data structure for autocomplete.",
    "Write a function to serialize and deserialize a binary tree.",
    "Implement Dijkstra's shortest path algorithm.",
    "Write a Python generator that yields Fibonacci numbers.",
    "Implement a thread-safe singleton pattern in Python.",
    "Write a function to detect cycles in a directed graph.",
    "Implement LRU cache from scratch.",
    "Write a Python function for binary exponentiation.",
    "Implement a producer-consumer pattern using asyncio queues.",
]

HELD_OUT_LAW = [
    "Explain the concept of vicarious liability in employment law.",
    "What is the difference between libel and slander?",
    "Define the doctrine of forum non conveniens.",
    "What are the requirements for a valid trust?",
    "Explain the exclusionary rule in criminal procedure.",
    "What is the difference between murder and manslaughter?",
    "Define the concept of eminent domain.",
    "Explain the doctrine of unclean hands in equity.",
    "What is the Business Judgment Rule in corporate law?",
    "Define the principle of double jeopardy.",
]


class ActivationCollector:
    """
    Registers forward pre-hooks to capture per-head and per-neuron activations.

    Attention: pre-hook on o_proj. Input shape [batch, q_len, head_dim * num_heads].
    MLP: pre-hook on down_proj. Input shape [batch, q_len, intermediate_size].

    For each prompt, captures a single L2 norm per unit (averaged over sequence length).
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


def compute_prototypes(model, tokenizer, coding_prompts, law_prompts):
    """
    Compute domain prototype embeddings by averaging the token embedding
    (embedding layer) across all positions and all prompts.

    Using the embedding layer's mean pool is fast (no transformer computation)
    and works as a bag-of-embeddings representation for domain classification.

    Returns:
        coding_proto: torch.tensor [hidden_size]
        law_proto: torch.tensor [hidden_size]
    """
    def get_emb(prompts):
        embs = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb = model.model.embed_tokens(inputs["input_ids"])[0]  # [seq_len, hidden]
                mean_emb = emb.mean(dim=0).cpu()
            embs.append(mean_emb)
        return torch.stack(embs).mean(dim=0)

    return get_emb(coding_prompts), get_emb(law_prompts)


def run_prompts(model, tokenizer, collector, prompts):
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)


def cohens_d(mean1, var1, n1, mean2, var2, n2):
    """
    Two-sample Cohen's d (pooled standard deviation).
    """
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return (mean1 - mean2) / (pooled_var.sqrt() + 1e-6)


def main():
    print("=" * 60)
    print("Differential Fidelity — Zero-Shot Subnetwork Profiling")
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold (Cohen's d): {THRESHOLD}")
    print(f"Train prompts: {len(TRAIN_CODING)} coding + {len(TRAIN_LAW)} law")
    print(f"Held-out prompts: {len(HELD_OUT_CODING)} coding + {len(HELD_OUT_LAW)} law")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Computing domain prototype embeddings...")
    coding_proto, law_proto = compute_prototypes(model, tokenizer, TRAIN_CODING, TRAIN_LAW)
    torch.save({"coding": coding_proto, "law": law_proto},
               os.path.join(OUTPUT_DIR, "domain_prototypes.pt"))
    print(f"  Prototype dim: {coding_proto.shape[0]}")

    print("[3/5] Registering activation hooks...")
    collector = ActivationCollector(model)

    print(f"[4/5] Profiling {len(TRAIN_CODING)} coding prompts...")
    run_prompts(model, tokenizer, collector, TRAIN_CODING)
    coding_attn = {k: torch.stack(v) for k, v in collector.attentions.items()}
    coding_mlp = {k: torch.stack(v) for k, v in collector.mlp_neurons.items()}
    collector.clear()

    print(f"[4/5] Profiling {len(TRAIN_LAW)} law prompts...")
    run_prompts(model, tokenizer, collector, TRAIN_LAW)
    law_attn = {k: torch.stack(v) for k, v in collector.attentions.items()}
    law_mlp = {k: torch.stack(v) for k, v in collector.mlp_neurons.items()}
    collector.remove_hooks()

    print("[5/5] Computing Cohen's d scores...")
    n_coding = len(TRAIN_CODING)
    n_law = len(TRAIN_LAW)

    law_heads = {}
    coding_heads = {}
    law_neuron_indices = {}
    coding_neuron_indices = {}

    for layer_idx in range(NUM_LAYERS):
        if layer_idx in coding_attn and layer_idx in law_attn:
            ca = coding_attn[layer_idx].squeeze(1)
            la = law_attn[layer_idx].squeeze(1)

            mean_c = ca.mean(dim=0)
            var_c = ca.var(dim=0, unbiased=False)
            mean_l = la.mean(dim=0)
            var_l = la.var(dim=0, unbiased=False)

            d = cohens_d(mean_l, var_l, n_law, mean_c, var_c, n_coding)
            l = torch.where(d > THRESHOLD)[0].tolist()
            c = torch.where(d < -THRESHOLD)[0].tolist()
            if l: law_heads[str(layer_idx)] = l
            if c: coding_heads[str(layer_idx)] = c

        if layer_idx in coding_mlp and layer_idx in law_mlp:
            cm = coding_mlp[layer_idx].squeeze(1)
            lm = law_mlp[layer_idx].squeeze(1)

            mean_c = cm.mean(dim=0)
            var_c = cm.var(dim=0, unbiased=False)
            mean_l = lm.mean(dim=0)
            var_l = lm.var(dim=0, unbiased=False)

            d = cohens_d(mean_l, var_l, n_law, mean_c, var_c, n_coding)
            l = torch.where(d > THRESHOLD)[0].tolist()
            c = torch.where(d < -THRESHOLD)[0].tolist()
            if l: law_neuron_indices[str(layer_idx)] = l
            if c: coding_neuron_indices[str(layer_idx)] = c

    mask = {
        "law_heads": law_heads,
        "coding_heads": coding_heads,
        "law_neurons": law_neuron_indices,
        "coding_neurons": coding_neuron_indices,
        "threshold": THRESHOLD,
        "metric": "cohens_d",
        "n_train_coding": n_coding,
        "n_train_law": n_law,
    }
    mask_path = os.path.join(OUTPUT_DIR, "mask.json")
    with open(mask_path, "w") as f:
        json.dump(mask, f, indent=2)

    held_out = {
        "coding": HELD_OUT_CODING,
        "law": HELD_OUT_LAW,
    }
    with open(os.path.join(OUTPUT_DIR, "held_out_prompts.json"), "w") as f:
        json.dump(held_out, f, indent=2)

    total_heads = NUM_LAYERS * NUM_HEADS
    total_neurons = NUM_LAYERS * INTERMEDIATE_SIZE
    total_params = 136_000_000

    def count_params(heads_dict, neurons_dict):
        h = sum(len(v) for v in heads_dict.values())
        n = sum(len(v) for v in neurons_dict.values())
        attn = 0
        for _, heads in heads_dict.items():
            attn += (len(heads) / NUM_HEADS) * HIDDEN_SIZE * (HIDDEN_SIZE + (NUM_KV_HEADS * HEAD_DIM) * 2 + HIDDEN_SIZE)
        mlp = n * HIDDEN_SIZE * 3
        return h, n, attn + mlp

    lh, ln, lp = count_params(law_heads, law_neuron_indices)
    ch, cn, cp = count_params(coding_heads, coding_neuron_indices)

    print("\n" + "=" * 60)
    print(f"Law-dominant heads:    {lh:>4} / {total_heads}  (params: {lp/1e6:.1f}M / {total_params/1e6:.0f}M = {100*lp/total_params:.1f}%)")
    print(f"Coding-dominant heads: {ch:>4} / {total_heads}  (params: {cp/1e6:.1f}M / {total_params/1e6:.0f}M = {100*cp/total_params:.1f}%)")
    print(f"Law-dominant neurons:    {ln:>5} / {total_neurons}  ({100*ln/total_neurons:.1f}%)")
    print(f"Coding-dominant neurons: {cn:>5} / {total_neurons}  ({100*cn/total_neurons:.1f}%)")
    print(f"Shared params (always BF16): {(total_params - lp - cp)/1e6:.1f}M / {total_params/1e6:.0f}M = {100*(total_params-lp-cp)/total_params:.1f}%")
    print(f"Mask saved to: {mask_path}")
    print(f"Prototypes saved to: {OUTPUT_DIR}/domain_prototypes.pt")
    print(f"Held-out prompts saved to: {OUTPUT_DIR}/held_out_prompts.json")
    print("=" * 60)

    for label, hd, nd in [("Law-dominant heads", law_heads, None),
                           ("Coding-dominant heads", coding_heads, None),
                           ("Law-dominant neurons", None, law_neuron_indices),
                           ("Coding-dominant neurons", None, coding_neuron_indices)]:
        if hd:
            print(f"\n{label}:")
            for lid, items in sorted(hd.items(), key=lambda x: int(x[0])):
                print(f"  Layer {lid}: {items}")
        if nd:
            print(f"\n{label}:")
            for lid, items in sorted(nd.items(), key=lambda x: int(x[0])):
                print(f"  Layer {lid}: {len(items)} neurons")


if __name__ == "__main__":
    main()
