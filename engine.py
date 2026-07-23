#!/usr/bin/env python3
"""
engine.py — Differential fidelity inference engine.

Router: embedding cosine similarity to domain prototypes.
Patching: surgical per-unit reconstruction from int8 compressed components.
          Only patch the opposing domain's tagged MLP neurons.
          All attention heads and shared weights stay full BF16.

Profiles:
  CODING: patch "law"-tagged MLP neurons → int8 (low fidelity)
  LAW:    patch "coding"-tagged MLP neurons → int8 (low fidelity)
  GENERAL: no patching (all BF16)
"""

import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MASK_PATH = "./outputs/mask.json"
COMPRESSED_DIR = "./compressed_weights"
PROTOTYPES_PATH = "./outputs/domain_prototypes.pt"
DTYPE = torch.bfloat16
DEVICE = "cpu"

HEAD_DIM = 64
HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536


class FidelityProfile(Enum):
    CODING = "coding"
    LAW = "law"
    GENERAL = "general"


class DifferentialEngine:
    """
    Context-aware inference engine with per-unit fidelity switching.

    Caches full BF16 state dict. Loads per-unit int8 compressed
    sidecars. On profile switch, surgically patches ONLY the MLP
    neurons tagged with the opposing domain's direction tag.
    All attention heads stay at full BF16.
    """

    def __init__(self, model_name=MODEL_NAME, compressed_dir=COMPRESSED_DIR,
                 mask_path=MASK_PATH, prototypes_path=PROTOTYPES_PATH):
        self.model_name = model_name
        self.compressed_dir = compressed_dir
        self.current_profile = None

        prototypes = torch.load(prototypes_path, map_location="cpu", weights_only=True)
        self.coding_prototype = prototypes["coding"]
        self.law_prototype = prototypes["law"]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=DTYPE, trust_remote_code=True
        ).to(DEVICE)
        self.model.eval()

        self.full_sd = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.compressed = {}
        manifest_path = os.path.join(compressed_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        for key, info in manifest.items():
            filepath = os.path.join(compressed_dir, info["file"])
            self.compressed[key] = torch.load(filepath, map_location=DEVICE, weights_only=False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def profile_from_prompt(self, prompt: str) -> FidelityProfile:
        """Classify prompt via embedding cosine similarity to domain prototypes."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = self.model.model.embed_tokens(inputs["input_ids"])[0]
            prompt_emb = emb.mean(dim=0).cpu()

        coding_sim = torch.cosine_similarity(
            prompt_emb.unsqueeze(0), self.coding_prototype.unsqueeze(0)
        ).item()
        law_sim = torch.cosine_similarity(
            prompt_emb.unsqueeze(0), self.law_prototype.unsqueeze(0)
        ).item()

        if abs(coding_sim - law_sim) < 0.02:
            return FidelityProfile.GENERAL
        return FidelityProfile.LAW if law_sim > coding_sim else FidelityProfile.CODING

    # ── Decompression ─────────────────────────────────────────────

    def _reconstruct_quant4_neuron(self, data, scale):
        return (data.float() * scale).to(DTYPE)

    # ── Profile loading ───────────────────────────────────────────

    def load_profile(self, profile: FidelityProfile):
        if profile == self.current_profile:
            return

        new_sd = dict(self.full_sd)

        if profile == FidelityProfile.CODING:
            target_tag = "law"  # patch law-tagged units
        elif profile == FidelityProfile.LAW:
            target_tag = "coding"  # patch coding-tagged units
        else:
            self.model.load_state_dict(new_sd, strict=False)
            self.current_profile = profile
            self._print_mem(profile)
            return

        for key, comp in self.compressed.items():
            if key not in new_sd:
                continue
            unit_tags = comp.get("unit_tags", {})
            to_patch = [int(k) for k, v in unit_tags.items() if v == target_tag]
            if not to_patch:
                continue

            if comp["type"] == "mlp":
                self._patch_mlp(new_sd[key], comp, to_patch)

        self.model.load_state_dict(new_sd, strict=False)
        self.current_profile = profile
        self._print_mem(profile)

    def _print_mem(self, profile):
        mem_bf16 = sum(v.numel() * 2 for v in self.full_sd.values() if v.dtype == DTYPE)
        mem_loaded = sum(
            v.numel() * 2 for v in self.model.state_dict().values() if v.dtype == DTYPE
        )
        saved = (mem_bf16 - mem_loaded) / (1024**2)
        print(f"\n[Profile: {profile.value.upper()}]")
        print(f"  Model memory (BF16 eq): {mem_loaded / (1024**3):.3f} GB  (saved {saved:.1f} MB)")
        if profile == FidelityProfile.CODING:
            print("  Patching: law-tagged MLP neurons → 4-bit  ·  Shared + coding attn: BF16")
        elif profile == FidelityProfile.LAW:
            print("  Patching: coding-tagged MLP neurons → 4-bit  ·  Shared + law attn: BF16")
        else:
            print("  All units: BF16")

    # ── Surgical patching ─────────────────────────────────────────

    def _patch_mlp(self, w_matrix, comp, patch_indices):
        q = comp["quant8"]
        idx_map = {q["neuron_indices"][i]: i for i in range(len(q["neuron_indices"]))}
        is_down = comp.get("is_down_proj", False)

        for n_idx in patch_indices:
            if n_idx not in idx_map:
                continue
            i = idx_map[n_idx]
            w_neuron = self._reconstruct_quant4_neuron(q["data"][i], q["scales"][i])
            if is_down:
                w_matrix[:, n_idx] = w_neuron
            else:
                w_matrix[n_idx, :] = w_neuron

    # ── Generation ─────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs):
        profile = self.profile_from_prompt(prompt)
        self.load_profile(profile)

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if raw.startswith(formatted):
            return raw[len(formatted):].strip()
        return raw.strip()


def print_banner():
    print(r"""
    ╔══════════════════════════════════════════════╗
    ║  Differential Fidelity Inference Engine      ║
    ║  Dual-domain tagging · Shared protection     ║
    ╚══════════════════════════════════════════════╝
    """)


def demo_mode():
    print_banner()
    engine = DifferentialEngine()

    coding_prompt = "Write a Python function to implement quicksort."
    law_prompt = "What are the elements of negligence in tort law?"

    for label, prompt in [("CODING", coding_prompt), ("LAW", law_prompt)]:
        print(f"\n{'─' * 50}")
        print(f"PROMPT ({label}): {prompt}")
        t0 = time.time()
        output = engine.generate(prompt, max_new_tokens=20)
        elapsed = time.time() - t0
        print(f"Response: {output}")
        print(f"Time: {elapsed:.2f}s")
    print(f"\n{'=' * 50}")
    print("Demo complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Differential Fidelity Inference Engine"
    )
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--tokens", type=int, default=50, help="Max new tokens")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    args = parser.parse_args()

    if args.demo:
        demo_mode()
    elif args.prompt:
        print_banner()
        engine = DifferentialEngine()
        output = engine.generate(args.prompt, max_new_tokens=args.tokens)
        print(f"\nInput:  {args.prompt}")
        print(f"Output: {output}")
    else:
        demo_mode()


if __name__ == "__main__":
    main()
