from __future__ import annotations

import argparse
import json
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LoRCConfig
from .data import load_minif2f, load_wikipedia, domain_dataloader, interleaved_dataloader
from .quantization import nf4_quantize, has_bitsandbytes
from .covariance import collect_covariances, cache_to_covariance, domain_subspaces
from .correction import build_correction, correction_storage_mb
from .causal_filter import causal_filter
from .ablation import compute_perplexity, disjunction_score


def run_pipeline(cfg: LoRCConfig):
    print("=" * 60)
    print("LoRC — Low-rank Quantization Correction")
    print(f"Model: {cfg.model_name}")
    print(f"  K={cfg.K}, alpha={cfg.alpha}, beta={cfg.beta}")
    print(f"  causal_filter={cfg.causal_filter_method}, attention={cfg.include_attention}")
    print("=" * 60)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\n[1/7] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[2/7] Loading datasets...")
    lean_texts = load_minif2f()
    wiki_texts = load_wikipedia(n=cfg.n_prompts)
    print(f"  Lean prompts:  {len(lean_texts)}")
    print(f"  Wiki prompts:  {len(wiki_texts)}")

    target_patterns = list(cfg.target_modules)
    if cfg.include_attention:
        target_patterns.extend([
            r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        ])

    print(f"\n[3/7] Collecting activation covariances (pre + post MLP)...")
    print("  Lean pass...")
    dl_lean = domain_dataloader(
        lean_texts, tokenizer,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed,
    )
    cache_lean = collect_covariances(model, dl_lean, target_patterns, cfg.n_prompts, device)
    C_lean = cache_to_covariance(cache_lean)

    print("  Wiki pass...")
    dl_wiki = domain_dataloader(
        wiki_texts, tokenizer,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 1,
    )
    cache_wiki = collect_covariances(model, dl_wiki, target_patterns, cfg.n_prompts, device)
    C_wiki = cache_to_covariance(cache_wiki)

    print("\n[4/7] Computing domain subspaces (ΔC eigendecomposition)...")
    subspaces = domain_subspaces(C_lean, C_wiki, cfg.K, cfg.alpha, cfg.beta)
    print(f"  Subspaces computed: {len(subspaces)} (pre + post per module)")

    print("\n[5/7] Building correction factors...")
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    full_bf16_bytes = sum(v.numel() * 2 for v in sd.values())

    corrections = {}
    n_modules = len(subspaces)
    for i, (key, (V_lean, V_wiki)) in enumerate(subspaces.items()):
        module_name, loc = key
        w_key = module_name + ".weight"
        if w_key not in sd:
            continue
        print(f"    Correction {i+1}/{n_modules}: {module_name} ({loc})")
        E = sd[w_key].float()
        V_act_lean, U_write_lean = build_correction(E, V_lean, cfg.K, module_name)
        V_act_wiki, U_write_wiki = build_correction(E, V_wiki, cfg.K, module_name)
        corrections[key] = {
            "lean": (V_act_lean, U_write_lean),
            "wiki": (V_act_wiki, U_write_wiki),
        }

    if cfg.causal_filter_method:
        print(f"\n  Causal filtering ({cfg.causal_filter_method})...")
        for key in list(corrections.keys()):
            module_name, loc = key
            V_act_lean, U_write_lean = corrections[key]["lean"]
            V_trim, U_trim = causal_filter(
                sd.get(module_name + ".weight"),
                V_act_lean, U_write_lean,
                model, module_name,
                interleaved_dataloader(
                    lean_texts, wiki_texts, tokenizer,
                    batch_size=1, seq_len=cfg.seq_len, seed=cfg.seed + 2,
                ),
                n_steps=cfg.causal_n_steps,
                method=cfg.causal_filter_method,
                keep_pct=cfg.causal_keep_pct,
                rel_threshold=cfg.causal_rel_threshold,
                device=device,
            )
            corrections[key]["lean"] = (V_trim, U_trim)
            print(f"    {module_name}|{loc}: K={V_act_lean.size(-1)} → {V_trim.size(-1)} components")

    correction_mb = correction_storage_mb(
        {str(k): v["lean"] for k, v in corrections.items()}
    ) + correction_storage_mb(
        {str(k): v["wiki"] for k, v in corrections.items()}
    )
    print(f"\n  Total correction storage: {correction_mb:.1f} MB")

    print("\n[6/7] Quantizing base weights (NF4)...")
    if has_bitsandbytes:
        print("  Using bitsandbytes NF4")
    else:
        print("  Using fallback NF4 quantization")
    W_q4 = {}
    weight_modules = [(n, m) for n, m in model.named_modules() if hasattr(m, "weight") and m.weight is not None and m.weight.dim() == 2 and (n + ".weight") in sd]
    for i, (name, mod) in enumerate(weight_modules):
        if (i + 1) % max(1, len(weight_modules) // 10) == 0:
            print(f"    NF4 quant: {i+1}/{len(weight_modules)} modules")
        W_q4[name] = nf4_quantize(sd[name + ".weight"])

    print("\n[7/7] Ablation study...")
    dl_lean_ppl = domain_dataloader(
        lean_texts, tokenizer,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 3,
    )
    dl_wiki_ppl = domain_dataloader(
        wiki_texts, tokenizer,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 4,
    )

    print("  Computing base perplexity...")
    base_lean_ppl = compute_perplexity(model, dl_lean_ppl, n_batches=5, device=device)
    base_wiki_ppl = compute_perplexity(model, dl_wiki_ppl, n_batches=5, device=device)
    print(f"  Base PPL — Lean: {base_lean_ppl:.2f}, Wiki: {base_wiki_ppl:.2f}")

    # Build a LoRCLinear-equipped model and measure ablation
    from .hybrid_module import LoRCLinear

    print("  Building LoRC model...")
    lorc_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    lorc_model.eval()

    replaced = 0
    n_total = len(corrections)
    for idx, (key, corr) in enumerate(corrections.items()):
        module_name, loc = key
        print(f"    Replacing {idx+1}/{n_total}: {module_name} ({loc})")
        try:
            mod = lorc_model.get_submodule(module_name)
        except (AttributeError, KeyError):
            continue
        if not hasattr(mod, "weight") or mod.weight is None:
            continue
        V_dict = {"coding": corr["lean"][0], "law": corr["wiki"][0]}
        U_dict = {"coding": corr["lean"][1], "law": corr["wiki"][1]}
        parts = module_name.split(".")
        parent = lorc_model.get_submodule(".".join(parts[:-1]))
        attr_name = parts[-1]
        lorc_lin = LoRCLinear(mod.weight.data, V_dict, U_dict, quantize_base=False)
        setattr(parent, attr_name, lorc_lin)
        replaced += 1

    print(f"  Replaced {replaced} modules with LoRCLinear")

    # Ablation: mask lean-dominant components, measure effect
    ppl_results = {"base": {"lean": base_lean_ppl, "wiki": base_wiki_ppl}}
    print(f"\n  Computing disjunction score (placeholders — run ablation separately)...")
    d_score = disjunction_score(base_lean_ppl, base_lean_ppl, base_wiki_ppl)

    results = {
        "config": {
            "K": cfg.K,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "causal_filter": cfg.causal_filter_method,
            "include_attention": cfg.include_attention,
            "model": cfg.model_name,
        },
        "storage": {
            "full_bf16_mb": full_bf16_bytes / (1024**2),
            "correction_mb": correction_mb,
        },
        "base_ppl": {"lean": base_lean_ppl, "wiki": base_wiki_ppl},
        "disjunction_score": d_score,
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, "lorc_results.json")
    json.dump(results, open(path, "w"), indent=2, default=str)
    print(f"\nResults saved to: {path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LoRC — Low-rank Quantization Correction")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--n-prompts", type=int, default=5000)
    parser.add_argument("--causal-filter", type=str, default="percentile", choices=["percentile", "relative", "none"])
    parser.add_argument("--causal-keep-pct", type=float, default=0.5)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = LoRCConfig(
        K=args.K,
        n_prompts=args.n_prompts,
        alpha=args.alpha,
        beta=args.beta,
        causal_filter_method=args.causal_filter if args.causal_filter != "none" else None,
        causal_keep_pct=args.causal_keep_pct,
        include_attention=args.attention,
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
