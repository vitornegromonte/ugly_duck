from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LoRCConfig
from .data import load_minif2f, load_wikipedia, domain_dataloader, interleaved_dataloader
from .quantization import nf4_quantize, nf4_dequantize
from .covariance import collect_covariances, cache_to_covariance, domain_subspaces
from .correction import build_correction, correction_storage_mb
from .causal_filter import causal_filter
from .ablation import compute_perplexity, disjunction_score, ablate_and_measure
from .hybrid_module import LoRCLinear, set_profile


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

    print("\n[5/7] Building correction factors from quantization error...")
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    full_bf16_bytes = sum(v.numel() * 2 for v in sd.values())

    quant_error_cache: dict[str, torch.Tensor] = {}
    grouped_corrections: dict[str, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]] = defaultdict(
        lambda: {"lean": [], "wiki": []}
    )
    n_modules = len(subspaces)
    for i, (key, (V_lean, V_wiki)) in enumerate(subspaces.items()):
        module_name, loc = key
        w_key = module_name + ".weight"
        if w_key not in sd:
            continue
        print(f"    Correction {i+1}/{n_modules}: {module_name} ({loc})")

        if w_key not in quant_error_cache:
            W_full = sd[w_key].float()
            W_dequant = nf4_dequantize(nf4_quantize(sd[w_key])).float()
            quant_error_cache[w_key] = W_full - W_dequant
        E = quant_error_cache[w_key]

        V_act_lean, U_write_lean = build_correction(E, V_lean, loc, cfg.K, module_name)
        V_act_wiki, U_write_wiki = build_correction(E, V_wiki, loc, cfg.K, module_name)
        grouped_corrections[module_name]["lean"].append((V_act_lean, U_write_lean))
        grouped_corrections[module_name]["wiki"].append((V_act_wiki, U_write_wiki))

    if cfg.causal_filter_method:
        print(f"\n  Causal filtering ({cfg.causal_filter_method})...")
        for module_name, parts in grouped_corrections.items():
            filtered = []
            for V_act_lean, U_write_lean in parts["lean"]:
                V_trim, U_trim = causal_filter(
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
                print(f"    {module_name}: K={V_act_lean.size(-1)} → {V_trim.size(-1)} lean components")
                filtered.append((V_trim, U_trim))
            parts["lean"] = filtered

    print("\n[6/7] Assembling LoRC model (NF4 base + corrections)...")
    corrections_final: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    correction_mb = 0.0
    for module_name, parts in grouped_corrections.items():
        entry = {}
        for domain in ("lean", "wiki"):
            pieces = parts[domain]
            if not pieces:
                continue
            V_cat = torch.cat([v for v, u in pieces], dim=1)
            U_cat = torch.cat([u for v, u in pieces], dim=1)
            entry[domain] = (V_cat, U_cat)
            correction_mb += (
                V_cat.numel() * V_cat.element_size() + U_cat.numel() * U_cat.element_size()
            ) / (1024**2)
        corrections_final[module_name] = entry
    print(f"  Total correction storage: {correction_mb:.1f} MB")

    lorc_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    lorc_model.eval()

    replaced = 0
    for module_name, entry in corrections_final.items():
        try:
            mod = lorc_model.get_submodule(module_name)
        except (AttributeError, KeyError):
            continue
        if not hasattr(mod, "weight") or mod.weight is None:
            continue
        V_dict = {domain: v for domain, (v, u) in entry.items()}
        U_dict = {domain: u for domain, (v, u) in entry.items()}
        parts = module_name.split(".")
        parent = lorc_model.get_submodule(".".join(parts[:-1]))
        attr_name = parts[-1]
        lorc_lin = LoRCLinear(mod.weight.data, V_dict, U_dict, quantize_base=True)
        setattr(parent, attr_name, lorc_lin)
        replaced += 1
    print(f"  Replaced {replaced} modules with LoRCLinear")

    print("\n[7/7] Ablation study...")
    pad_id = tokenizer.pad_token_id

    dl_lean_ppl = domain_dataloader(
        lean_texts, tokenizer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 3,
    )
    dl_wiki_ppl = domain_dataloader(
        wiki_texts, tokenizer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 4,
    )
    print("  Computing base perplexity (full-precision model)...")
    base_lean_ppl = compute_perplexity(model, dl_lean_ppl, n_batches=5, device=device, pad_token_id=pad_id)
    base_wiki_ppl = compute_perplexity(model, dl_wiki_ppl, n_batches=5, device=device, pad_token_id=pad_id)
    print(f"  Base PPL — Lean: {base_lean_ppl:.2f}, Wiki: {base_wiki_ppl:.2f}")

    print("  Computing LoRC (quantized base + correction) perplexity...")
    dl_lean_lorc = domain_dataloader(
        lean_texts, tokenizer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 9,
    )
    dl_wiki_lorc = domain_dataloader(
        wiki_texts, tokenizer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 10,
    )
    set_profile(lorc_model, "lean")
    lorc_lean_ppl = compute_perplexity(lorc_model, dl_lean_lorc, n_batches=5, device=device, pad_token_id=pad_id)
    set_profile(lorc_model, "wiki")
    lorc_wiki_ppl = compute_perplexity(lorc_model, dl_wiki_lorc, n_batches=5, device=device, pad_token_id=pad_id)
    set_profile(lorc_model, None)
    print(f"  LoRC PPL  — Lean: {lorc_lean_ppl:.2f}, Wiki: {lorc_wiki_ppl:.2f}")

    print("  Ablating lean-dominant subspace components (causal disjunction test)...")
    dl_lean_ab1 = domain_dataloader(
        lean_texts, tokenizer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 5,
    )
    dl_wiki_ab1 = domain_dataloader(
        wiki_texts, tokenizer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 6,
    )
    lean_ab_lean_ppl, lean_ab_wiki_ppl = ablate_and_measure(
        model, subspaces, "lean", dl_lean_ab1, dl_wiki_ab1, n_batches=5, device=device, pad_token_id=pad_id,
    )
    print(f"    Masked-lean → Lean PPL: {lean_ab_lean_ppl:.2f}, Wiki PPL: {lean_ab_wiki_ppl:.2f}")

    print("  Ablating wiki-dominant subspace components...")
    dl_lean_ab2 = domain_dataloader(
        lean_texts, tokenizer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 7,
    )
    dl_wiki_ab2 = domain_dataloader(
        wiki_texts, tokenizer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed + 8,
    )
    wiki_ab_lean_ppl, wiki_ab_wiki_ppl = ablate_and_measure(
        model, subspaces, "wiki", dl_lean_ab2, dl_wiki_ab2, n_batches=5, device=device, pad_token_id=pad_id,
    )
    print(f"    Masked-wiki → Lean PPL: {wiki_ab_lean_ppl:.2f}, Wiki PPL: {wiki_ab_wiki_ppl:.2f}")

    d_score_lean = disjunction_score(base_lean_ppl, base_wiki_ppl, lean_ab_lean_ppl, lean_ab_wiki_ppl)
    d_score_wiki = disjunction_score(base_wiki_ppl, base_lean_ppl, wiki_ab_wiki_ppl, wiki_ab_lean_ppl)
    print(f"\n  Disjunction score (lean components): {d_score_lean:.3f}")
    print(f"  Disjunction score (wiki components): {d_score_wiki:.3f}")

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
        "lorc_ppl": {"lean": lorc_lean_ppl, "wiki": lorc_wiki_ppl},
        "ablation": {
            "mask_lean": {"lean_ppl": lean_ab_lean_ppl, "wiki_ppl": lean_ab_wiki_ppl},
            "mask_wiki": {"lean_ppl": wiki_ab_lean_ppl, "wiki_ppl": wiki_ab_wiki_ppl},
        },
        "disjunction_score": {"lean_components": d_score_lean, "wiki_components": d_score_wiki},
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
