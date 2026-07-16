#!/usr/bin/env python3
"""
build_compressed_weights.py — Build compressed sidecar weights.

For each weight tensor in the model:
  - If the owning layer has law-tagged heads or neurons, apply BINARY (1-bit)
    structural compression: sign() decomposition with per-row scaling.
  - Otherwise, apply Q4_K-style uniform quantization (4-bit, 16 levels).

The original model weights are preserved in full precision. Compressed
sidecars are saved to ./compressed_weights/ with a manifest that the
engine uses to load per-profile fidelities.

Binary compression math:
    w_binary = sign(w)              # {+1, -1}
    scale    = mean(|w|, dim=row)   # float32 per row
    w_hat    = w_binary * scale     # reconstruction (broadcast)

Q4 uniform quantization:
    delta  = (max - min) / 15       # step size per row
    w_q4   = round((w - min) / delta), clamped to [0, 15]
    w_hat  = w_q4 * delta + min     # reconstruction

Reference:
    "Zero-shot sub-network identification via subtractive probing (Cao et al.)"
    "Context-aware fidelity profile loading (Kim et al., Fan et al.)"
"""

import json
import math
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
HEAD_DIM = 64
INTERMEDIATE_SIZE = 1536
HIDDEN_SIZE = 576


def compress_binary(w: torch.Tensor, dim: int = 0):
    """
    Binary (1-bit) compression with per-row scaling.

    Storage: 1 bit/param + 32 bits per row (float32 scale).
    For [2048, 2048]: 0.5 MB + 8 KB ≈ 0.5 MB vs BF16 8 MB → ~16x.
    """
    w_binary = torch.sign(w)
    scale = w.abs().mean(dim=dim, keepdim=True)
    binary_uint8 = ((w_binary + 1) // 2).to(torch.uint8)
    return {
        "type": "binary",
        "binary": binary_uint8,
        "scale": scale.to(torch.float32),
        "shape": list(w.shape),
        "dim": dim,
    }


def compress_q4(w: torch.Tensor):
    """
    Q4_K-style uniform quantization (16 levels, 4-bit).

    Storage: 4 bits/param + 64 bits per row (float32 min + float32 delta).
    For [2048, 2048]: 2 MB + 16 KB ≈ 2 MB vs BF16 8 MB → ~4x.
    """
    min_val = w.min(dim=-1, keepdim=True).values
    max_val = w.max(dim=-1, keepdim=True).values
    delta = (max_val - min_val) / 15.0
    q = ((w - min_val) / delta.clamp(min=1e-10)).round().clamp(0, 15).to(torch.uint8)
    return {
        "type": "q4",
        "q4": q,
        "min": min_val.to(torch.float32),
        "delta": delta.to(torch.float32),
        "shape": list(w.shape),
    }


def tensor_key_to_layer_info(key: str):
    """
    Parse state_dict key -> (layer_idx, component_type).

    "model.layers.5.self_attn.q_proj.weight" -> (5, "attention")
    "model.layers.10.mlp.gate_proj.weight"  -> (10, "mlp")
    "model.embed_tokens.weight"              -> (-1, "other")
    """
    match = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp)\.", key)
    if match:
        return int(match.group(1)), match.group(2)
    return -1, "other"


def main():
    print("=" * 60)
    print("Differential Fidelity — Build Compressed Weights")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n[1/5] Loading mask...")
    with open(MASK_PATH) as f:
        mask = json.load(f)
    law_heads = mask.get("law_heads", {})
    law_neurons = mask.get("law_neurons", {})
    print(f"  Law-tagged layers (heads):   {len(law_heads)}")
    print(f"  Law-tagged layers (neurons): {len(law_neurons)}")

    print("[2/5] Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE)
    sd = model.state_dict()
    del model

    print("[3/5] Determining compression schemes...")
    weight_info = {}
    total_bf16_bytes = 0
    for key, tensor in sd.items():
        if not key.endswith(".weight") or tensor.ndim != 2:
            continue
        layer_idx, component = tensor_key_to_layer_info(key)
        is_law = (component == "attention" and str(layer_idx) in law_heads) or \
                 (component == "mlp" and str(layer_idx) in law_neurons)
        n_bytes = tensor.numel() * 2
        total_bf16_bytes += n_bytes
        weight_info[key] = {
            "layer": layer_idx, "component": component,
            "shape": tensor.shape, "is_law": is_law, "bf16_bytes": n_bytes,
        }

    print("[4/5] Compressing weights...")
    manifest = {}
    total_compressed_bytes = 0
    num_law = 0
    num_q4 = 0

    for key, info in weight_info.items():
        tensor = sd[key]
        if info["is_law"]:
            comp = compress_binary(tensor)
            tag = "law_binary"
            num_law += 1
        else:
            comp = compress_q4(tensor)
            tag = "coding_q4"
            num_q4 += 1

        safe_key = key.replace(".", "_").replace("/", "_")
        filename = f"{safe_key}.pt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        torch.save(comp, filepath)

        if tag == "law_binary":
            n_bits = tensor.numel()
            n_scale = tensor.shape[0] * 32
            compressed_bytes = math.ceil(n_bits / 8) + math.ceil(n_scale / 8)
        else:
            n_bits = tensor.numel() * 4
            n_meta = tensor.shape[0] * 32 * 2
            compressed_bytes = math.ceil(n_bits / 8) + math.ceil(n_meta / 8)
        total_compressed_bytes += compressed_bytes

        manifest[key] = {
            "file": filename, "tag": tag, "shape": info["shape"],
            "bf16_bytes": info["bf16_bytes"], "compressed_bytes": compressed_bytes,
        }

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("[5/5] Summary")
    print("\n" + "=" * 60)
    bf16_gb = total_bf16_bytes / (1024**3)
    compressed_gb = total_compressed_bytes / (1024**3)
    reduction_pct = (1 - compressed_gb / bf16_gb) * 100
    print(f"BF16 total size:        {bf16_gb:.3f} GB")
    print(f"Compressed total size:  {compressed_gb:.3f} GB")
    print(f"Storage reduction:      {reduction_pct:.1f}%")
    print(f"\n  Law-binary tensors: {num_law}")
    print(f"  Coding-Q4 tensors:  {num_q4}")
    print(f"\nManifest: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
