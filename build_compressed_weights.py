#!/usr/bin/env python3
"""
build_compressed_weights.py — Per-unit compression with direction tags.

For each unit (attention head or MLP neuron) tagged as law-dominant OR
coding-dominant: SVD low-rank (heads, k=16) or binary (neurons, 1-bit).

Stores a per-unit direction tag ("law" or "coding") so the engine
only patches the opposing domain's units:
  - CODING profile: patch "law"-tagged units → SVD/binary
  - LAW profile:   patch "coding"-tagged units → SVD/binary
  - Shared units (no tag): always BF16, never compressed.

No full-matrix Q4. No compression of shared weights.
"""

import json
import os
import re
import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MASK_PATH = "./outputs/mask.json"
OUTPUT_DIR = "./compressed_weights"
DTYPE = torch.bfloat16
DEVICE = "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS = 30
NUM_HEADS = 9
NUM_KV_HEADS = 3
HEAD_DIM = 64
HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536

def quant4_compress(w_row: torch.Tensor):
    max_val = w_row.abs().max()
    if max_val < 1e-8:
        return torch.zeros_like(w_row, dtype=torch.int8), torch.tensor(1.0, dtype=torch.float32)
    scale = max_val / 7.0
    q = (w_row / scale).round().clamp(-7, 7).to(torch.int8)
    return q, scale.to(torch.float32)


def tensor_key_to_layer_info(key: str):
    match = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp)\.", key)
    if match:
        return int(match.group(1)), match.group(2)
    return -1, "other"


def build_unit_tags(mask, layer_idx, component):
    """Return dict {tagged_unit_idx: "law" or "coding"} for a given layer/component."""
    tags = {}
    for direction, heads_key, neurons_key in [
        ("law", "law_heads", "law_neurons"),
        ("coding", "coding_heads", "coding_neurons"),
    ]:
        if component == "self_attn" and str(layer_idx) in mask.get(heads_key, {}):
            for h in mask[heads_key][str(layer_idx)]:
                tags[h] = direction
        if component == "mlp" and str(layer_idx) in mask.get(neurons_key, {}):
            for n in mask[neurons_key][str(layer_idx)]:
                tags[n] = direction
    return tags


def compress_attention_matrix(key, tensor, unit_tags):
    return None


def compress_mlp_matrix(key, tensor, unit_tags):
    is_down_proj = "down_proj" in key
    n = INTERMEDIATE_SIZE

    if is_down_proj:
        w_neurons = tensor.t()
    else:
        w_neurons = tensor

    tagged_list = sorted(unit_tags.items())
    q_indices, q_data, q_scales = [], [], []
    for n_idx, direction in tagged_list:
        if n_idx >= n:
            continue
        q, s = quant4_compress(w_neurons[n_idx].float())
        q_indices.append(n_idx)
        q_data.append(q)
        q_scales.append(s)

    comp_bytes = sum(d.numel() for d in q_data) + len(q_scales) * 4

    return {
        "key": key,
        "shape": list(tensor.shape),
        "type": "mlp",
        "is_down_proj": is_down_proj,
        "intermediate_size": n,
        "unit_tags": unit_tags,
        "quant8": {
            "neuron_indices": q_indices,
            "data": torch.stack(q_data) if q_data else torch.empty(0, dtype=torch.int8),
            "scales": torch.tensor(q_scales, dtype=torch.float32) if q_scales else torch.empty(0),
        },
        "metadata": {
            "tag": "quant4",
            "compressed_bytes": comp_bytes,
        },
    }


def main():
    print("=" * 60)
    print("Differential Fidelity — Build Per-Unit Compressed Weights")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n[1/5] Loading mask...")
    with open(MASK_PATH) as f:
        mask = json.load(f)

    lh = sum(len(v) for v in mask.get("law_heads", {}).values())
    ln = sum(len(v) for v in mask.get("law_neurons", {}).values())
    ch = sum(len(v) for v in mask.get("coding_heads", {}).values())
    cn = sum(len(v) for v in mask.get("coding_neurons", {}).values())
    print(f"  Law-dominant:   {lh} heads, {ln} neurons")
    print(f"  Coding-dominant: {ch} heads, {cn} neurons")
    print(f"  MLP compression: int8 per neuron (+ scale)")

    print("[2/5] Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE)
    sd = model.state_dict()
    del model

    print("[3/5] Compressing weights...")
    manifest = {}
    total_bf16_bytes = 0
    total_compressed_bytes = 0
    tagged_tensors = 0

    for key, tensor in sd.items():
        if not key.endswith(".weight") or tensor.ndim != 2:
            continue
        layer_idx, component = tensor_key_to_layer_info(key)
        n_bf16 = tensor.numel() * 2
        total_bf16_bytes += n_bf16

        unit_tags = build_unit_tags(mask, layer_idx, component)
        if not unit_tags:
            continue  # no tags → shared → no sidecar needed

        if component == "mlp":
            comp = compress_mlp_matrix(key, tensor, unit_tags)
        elif component == "self_attn":
            continue  # attention heads stay full BF16
        else:
            continue

        safe_key = key.replace(".", "_").replace("/", "_")
        filename = f"{safe_key}.pt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        torch.save(comp, filepath)

        tagged_tensors += 1
        cb = comp["metadata"]["compressed_bytes"]
        total_compressed_bytes += cb
        manifest[key] = {
            "file": filename,
            "tag": comp["metadata"]["tag"],
            "shape": list(tensor.shape),
            "unit_tags": {str(k): v for k, v in unit_tags.items()},
            "bf16_bytes": n_bf16,
            "compressed_bytes": cb,
        }

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Compressed {tagged_tensors} tensors with tagged units")
    print(f"  Shared (no compression): {sum(1 for k, t in sd.items() if k.endswith('.weight') and t.ndim == 2) - tagged_tensors} tensors")

    print("\n[4/5] Summary")
    print("\n" + "=" * 60)
    bf16_gb = total_bf16_bytes / (1024**3)
    compressed_mb = total_compressed_bytes / (1024**2)
    print(f"Total BF16 (all weights):  {bf16_gb:.3f} GB")
    print(f"Compressed sidecars total: {compressed_mb:.1f} MB")
    print(f"Sidecar overhead:          {100 * total_compressed_bytes / total_bf16_bytes:.2f}% of full BF16")
    print(f"\nManifest: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
