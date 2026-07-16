#!/usr/bin/env python3
"""
engine.py — Native context-aware inference engine with profile loading.

The engine uses a keyword-based prompt classifier to determine the domain
(coding, law, or general), then loads the appropriate fidelity profile:
  - CODING: law regions at binary (1-bit), coding/shared at full BF16
  - LAW: law regions at full BF16, coding regions at Q4 (4-bit)
  - GENERAL: all weights at full BF16 (no compression)

Weight swapping is done via model.load_state_dict() with a mixed state dict.

DESIGN NOTE (keyword classifier):
    A production system would use a small BERT classifier or the model's own
    prefill activations as a router. For the weekend demo, keyword matching
    is transparent, debuggable, and zero-overhead.

Reference:
    Kim et al., Fan et al. — Context-aware fidelity profile loading inspired
    by MoE offloading strategies.
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
DTYPE = torch.bfloat16
DEVICE = "cpu"

LAW_KEYWORDS = [
    "contract", "tort", "statute", "precedent", "negligence",
    "fiduciary", "stare decisis", "breach", "liability", "damages",
]
CODING_KEYWORDS = [
    "python", "function", "algorithm", "math", "sort",
    "recursive", "implementation", "array", "graph", "tree",
]


class FidelityProfile(Enum):
    CODING = "coding"
    LAW = "law"
    GENERAL = "general"


class DifferentialEngine:
    """
    Context-aware inference engine with per-profile weight fidelity.

    Keeps a cached full-BF16 state dict and a dict of compressed sidecars.
    On profile switch, builds a new state dict with the right mix of
    decompressed and full-precision tensors, then calls load_state_dict.
    """

    def __init__(self, model_name=MODEL_NAME, compressed_dir=COMPRESSED_DIR,
                 mask_path=MASK_PATH):
        self.model_name = model_name
        self.compressed_dir = compressed_dir
        self.mask_path = mask_path
        self.current_profile = None

        with open(mask_path) as f:
            self.mask = json.load(f)
        manifest_path = os.path.join(compressed_dir, "manifest.json")
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=DTYPE,
            trust_remote_code=True,
        ).to(DEVICE)
        self.model.eval()

        self.full_sd = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.compressed_weights = {}
        for key, info in self.manifest.items():
            filepath = os.path.join(compressed_dir, info["file"])
            self.compressed_weights[key] = torch.load(
                filepath, map_location=DEVICE, weights_only=False
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def profile_from_prompt(self, prompt: str) -> FidelityProfile:
        """
        Classify prompt into CODING, LAW, or GENERAL using keyword matching.
        """
        prompt_lower = prompt.lower()
        law_score = sum(1 for kw in LAW_KEYWORDS if kw in prompt_lower)
        coding_score = sum(1 for kw in CODING_KEYWORDS if kw in prompt_lower)

        if law_score > coding_score and law_score > 0:
            return FidelityProfile.LAW
        elif coding_score > law_score and coding_score > 0:
            return FidelityProfile.CODING
        return FidelityProfile.GENERAL

    def _decompress_binary(self, comp: dict) -> torch.Tensor:
        """Decompress binary-coded weight back to BF16."""
        binary_int = comp["binary"].to(torch.int8) * 2 - 1
        scale = comp["scale"]
        w = binary_int * scale
        return w.to(DTYPE)

    def _decompress_q4(self, comp: dict) -> torch.Tensor:
        """Decompress Q4-coded weight back to BF16."""
        q4 = comp["q4"]
        min_val = comp["min"]
        delta = comp["delta"]
        w = q4.to(DTYPE) * delta + min_val
        return w.to(DTYPE)

    def load_profile(self, profile: FidelityProfile):
        """Build and load a state dict matching the requested profile."""
        if profile == self.current_profile:
            return

        new_sd = dict(self.full_sd)

        if profile == FidelityProfile.CODING:
            for key, comp in self.compressed_weights.items():
                if comp["type"] == "binary" and key in new_sd:
                    new_sd[key] = self._decompress_binary(comp)

        elif profile == FidelityProfile.LAW:
            for key, comp in self.compressed_weights.items():
                if comp["type"] == "q4" and key in new_sd:
                    new_sd[key] = self._decompress_q4(comp)

        self.model.load_state_dict(new_sd, strict=False)
        self.current_profile = profile

        mem_bf16 = sum(v.numel() * 2 for v in self.full_sd.values() if v.dtype == DTYPE)
        mem_loaded = sum(
            v.numel() * 2 for v in self.model.state_dict().values() if v.dtype == DTYPE
        )
        saved = (mem_bf16 - mem_loaded) / (1024**2)

        print(f"\n[Profile: {profile.value.upper()}]")
        print(f"  Model memory (BF16 equivalent): {mem_loaded / (1024**3):.3f} GB"
              f"  (saved {saved:.1f} MB vs full BF16)")
        if profile == FidelityProfile.CODING:
            print("  Law regions: 1-bit binary  ·  Coding regions: BF16")
        elif profile == FidelityProfile.LAW:
            print("  Law regions: BF16  ·  Coding regions: 4-bit Q4")
        else:
            print("  All regions: BF16")

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs):
        """Generate text using the profile inferred from the prompt."""
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
    ║  Zero-shot subnetworks · Binary compression  ║
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
        print(f"{'─' * 50}")
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
