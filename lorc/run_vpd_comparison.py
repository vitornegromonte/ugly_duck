from __future__ import annotations

import argparse
import json
import torch
from torch import Tensor

from .config import LoRCConfig
from .ablation import component_overlap


def load_vpd_run(path: str) -> dict[str, Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    read_vectors: dict[str, Tensor] = {}
    for key, components in data.items():
        if isinstance(components, dict) and "V" in components and "U" in components:
            read_vectors[key] = components["V"]
    return read_vectors


def compare(model_name: str, vpd_path: str, output_dir: str = "./results"):
    print("=" * 60)
    print("LoRC vs VPD — Component Subspace Comparison")
    print(f"Model: {model_name}")
    print(f"VPD run: {vpd_path}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[1/3] Loading VPD components...")
    vpd_components = load_vpd_run(vpd_path)
    print(f"  Loaded {len(vpd_components)} VPD modules")

    print("\n[2/3] Loading LoRC corrections...")
    from .run_pipeline import run_pipeline
    cfg = LoRCConfig(model_name=model_name, output_dir=output_dir, device=device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import load_minif2f, load_wikipedia, interleaved_dataloader
    from .covariance import collect_covariances, domain_subspaces
    from .correction import build_correction

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lean_texts = load_minif2f()
    wiki_texts = load_wikipedia(n=cfg.n_prompts)

    dataloader = interleaved_dataloader(
        lean_texts, wiki_texts, tokenizer,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed,
    )
    pre_cache, _ = collect_covariances(
        model, dataloader, cfg.target_modules, cfg.n_prompts, device=device
    )
    C_lean_pre, C_wiki_pre = {}, {}
    for (name, loc), (cov, _) in pre_cache.items():
        C_lean_pre[(name, loc)] = cov

    dataloader = interleaved_dataloader(
        wiki_texts, lean_texts, tokenizer,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 1,
    )
    pre_cache_w, _ = collect_covariances(
        model, dataloader, cfg.target_modules, cfg.n_prompts, device=device
    )
    for (name, loc), (cov, _) in pre_cache_w.items():
        C_wiki_pre[(name, loc)] = cov

    subspaces = domain_subspaces(C_lean_pre, C_wiki_pre, cfg.K, cfg.alpha, cfg.beta)

    sd = model.state_dict()
    lorc_components: dict[str, Tensor] = {}
    for key, (V_lean, V_wiki) in subspaces.items():
        module_name, loc = key
        w_key = module_name + ".weight"
        if w_key not in sd:
            continue
        E = sd[w_key].float()
        V_act_lean, _ = build_correction(E, V_lean, cfg.K)
        V_act_wiki, _ = build_correction(E, V_wiki, cfg.K)
        lorc_components[module_name + "_lean"] = V_act_lean
        lorc_components[module_name + "_wiki"] = V_act_wiki

    print(f"\n[3/3] Computing cosine similarity between subspaces...")
    similarities: dict[str, float] = {}
    for lorc_key, V_lorc in lorc_components.items():
        module_key = lorc_key.rsplit("_", 1)[0]
        if module_key not in vpd_components:
            continue
        V_vpd = vpd_components[module_key]
        if V_lorc.dim() != 2 or V_vpd.dim() != 2:
            continue
        sim = component_overlap(V_lorc, V_vpd)
        max_sim = sim.abs().max().item()
        similarities[lorc_key] = max_sim
        print(f"  {lorc_key:<50} max cosine sim: {max_sim:.4f}")

    results = {
        "n_lorc_components": len(lorc_components),
        "n_vpd_modules": len(vpd_components),
        "similarities": similarities,
        "mean_similarity": sum(similarities.values()) / max(len(similarities), 1),
    }
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "vpd_comparison.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LoRC vs VPD Comparison")
    parser.add_argument("--vpd-path", type=str, required=True, help="Path to VPD checkpoint")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    compare(args.model, args.vpd_path, args.output_dir)


if __name__ == "__main__":
    main()
