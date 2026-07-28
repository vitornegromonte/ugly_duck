from __future__ import annotations

"""
lean_eval.py — Verified disjunction test: does masking lean-dominant subspace
components collapse Lean *proof-checking* pass rate more than masking
wiki-dominant components does, and vice versa?

Unlike ablation.compute_perplexity (a proxy), this scores generations with
the actual Lean 4 compiler on minif2f, so "correct" means "the kernel
accepted the proof," not "low perplexity."

Requires a local Lean 4 / Lake project with Mathlib already built, e.g.:

    git clone https://github.com/leanprover-community/mathlib4
    cd mathlib4 && lake exe cache get

Usage:
    python -m lorc.lean_eval --lean-project /path/to/mathlib4 --n-problems 20
"""

import argparse
import json
import os
import re
import time

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ablation import ablated_subspace
from .covariance import cache_to_covariance, collect_covariances, domain_subspaces
from .data import domain_dataloader, load_wikipedia
from .lean_verifier import check_lean_toolchain, pass_at_k, split_statement, verify_proof

TARGET_MODULES = [r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)"]


def build_prompt(header: str, statement_stub: str) -> str:
    return (
        "Complete the following Lean 4 proof. Output only the proof term or "
        "tactic block that replaces `sorry` -- no explanation, no markdown fences.\n\n"
        f"{header}\n\n{statement_stub}\n"
    )


def extract_proof_body(generated: str) -> str:
    text = generated.strip()
    text = re.sub(r"^```(?:lean4?)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    first_stub_break = re.search(r"\n(theorem|lemma|example)\b", text)
    if first_stub_break:
        text = text[: first_stub_break.start()].strip()
    return text if text else "sorry"


@torch.no_grad()
def generate_candidate(
    model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
    return extract_proof_body(text)


def evaluate_condition(
    model, tokenizer, items, n_samples, max_new_tokens, temperature, lean_project_dir, scratch_subdir, device, label,
):
    results = []
    for item in items:
        stub = split_statement(item["formal_statement"])
        prompt = build_prompt(item["header"], stub)
        n_correct = 0
        for _ in range(n_samples):
            proof_body = generate_candidate(model, tokenizer, prompt, max_new_tokens, temperature, device)
            result = verify_proof(item["header"], stub, proof_body, lean_project_dir, scratch_subdir)
            n_correct += int(result.passed)
        results.append({"id": item["id"], "n": n_samples, "c": n_correct})
        print(f"    [{label}] {item['id']}: {n_correct}/{n_samples} passed")
    return results


def summarize(results: list[dict], k: int) -> float:
    if not results:
        return 0.0
    scores = [pass_at_k(r["n"], r["c"], k) for r in results]
    return sum(scores) / len(scores)


def build_summary(
    args: argparse.Namespace,
    n_problems: int,
    base_pk: float,
    mask_lean_pk: float,
    mask_wiki_pk: float,
    drop_lean: float,
    drop_wiki: float,
    verified_disjunction_score: float,
    started_at: float,
) -> dict:
    return {
        "model": args.model,
        "lean_project": args.lean_project,
        "n_problems": n_problems,
        "n_samples_per_problem": args.n_samples,
        "k": args.k,
        "K_subspace": args.K,
        "alpha": args.alpha,
        "beta": args.beta,
        "n_cov_prompts": args.n_cov_prompts,
        "pass_at_k": {"base": base_pk, "mask_lean": mask_lean_pk, "mask_wiki": mask_wiki_pk},
        "pass_rate_drop": {"mask_lean": drop_lean, "mask_wiki": drop_wiki},
        "verified_disjunction_score": verified_disjunction_score,
        "elapsed_seconds": round(time.time() - started_at, 1),
    }


def save_summary(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="LoRC — Lean-verified disjunction eval")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--lean-project", required=True, help="Path to a Lake project with Mathlib built")
    parser.add_argument("--lean-scratch-subdir", default="LorcScratch")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--n-cov-prompts", type=int, default=500)
    parser.add_argument("--n-problems", type=int, default=20, help="minif2f validation problems to evaluate")
    parser.add_argument("--n-samples", type=int, default=4, help="samples per problem, for pass@k")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output", default="./results/lean_disjunction.json", help="Full per-problem results")
    parser.add_argument(
        "--summary-output", default="./results/lean_disjunction_summary.json",
        help="Concise aggregate-only summary (no per-problem detail)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started_at = time.time()
    check_lean_toolchain(args.lean_project)
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("LoRC — Lean-Verified Disjunction Test")
    print(f"Model: {args.model}")
    print(f"Lean project: {args.lean_project}")
    print("=" * 60)

    print("\n[1/5] Loading model + tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Loading minif2f (validation split, with header)...")
    ds = datasets.load_dataset("cat-searcher/minif2f-lean4", split="validation")
    all_items = [ex for ex in ds if ex.get("formal_statement") and ex.get("header")]
    items = all_items[: args.n_problems]
    lean_texts = [ex["formal_statement"] for ex in all_items]
    print(f"  Evaluating on {len(items)}/{len(all_items)} problems")

    print("  Loading wikitext (contrastive domain for covariance)...")
    wiki_texts = load_wikipedia(n=args.n_cov_prompts)

    print("\n[3/5] Collecting activation covariances...")
    dl_lean = domain_dataloader(lean_texts, tokenizer, batch_size=8, seq_len=args.seq_len, seed=args.seed)
    cache_lean = collect_covariances(model, dl_lean, TARGET_MODULES, args.n_cov_prompts, device)
    C_lean = cache_to_covariance(cache_lean)

    dl_wiki = domain_dataloader(wiki_texts, tokenizer, batch_size=8, seq_len=args.seq_len, seed=args.seed + 1)
    cache_wiki = collect_covariances(model, dl_wiki, TARGET_MODULES, args.n_cov_prompts, device)
    C_wiki = cache_to_covariance(cache_wiki)

    print("[4/5] Computing domain subspaces (ΔC eigendecomposition)...")
    subspaces = domain_subspaces(C_lean, C_wiki, args.K, args.alpha, args.beta)

    print(f"\n[5/5] Generating + verifying proofs ({args.n_samples} samples/problem)...")
    print("\n  Condition: BASE (no ablation)")
    base_results = evaluate_condition(
        model, tokenizer, items, args.n_samples, args.max_new_tokens, args.temperature,
        args.lean_project, args.lean_scratch_subdir, device, "base",
    )

    print("\n  Condition: MASK_LEAN (lean-dominant components projected out)")
    with ablated_subspace(model, subspaces, "lean"):
        mask_lean_results = evaluate_condition(
            model, tokenizer, items, args.n_samples, args.max_new_tokens, args.temperature,
            args.lean_project, args.lean_scratch_subdir, device, "mask_lean",
        )

    print("\n  Condition: MASK_WIKI (wiki-dominant components projected out)")
    with ablated_subspace(model, subspaces, "wiki"):
        mask_wiki_results = evaluate_condition(
            model, tokenizer, items, args.n_samples, args.max_new_tokens, args.temperature,
            args.lean_project, args.lean_scratch_subdir, device, "mask_wiki",
        )

    base_pk = summarize(base_results, args.k)
    mask_lean_pk = summarize(mask_lean_results, args.k)
    mask_wiki_pk = summarize(mask_wiki_results, args.k)

    drop_lean = max(base_pk - mask_lean_pk, 1e-8)
    drop_wiki = max(base_pk - mask_wiki_pk, 1e-8)
    verified_disjunction_score = drop_lean / drop_wiki

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"pass@{args.k}  base={base_pk:.3f}  mask_lean={mask_lean_pk:.3f}  mask_wiki={mask_wiki_pk:.3f}")
    print(f"Pass-rate drop from masking lean components: {drop_lean:.3f}")
    print(f"Pass-rate drop from masking wiki components: {drop_wiki:.3f}")
    print(f"Verified disjunction score (>1 = genuine domain separation): {verified_disjunction_score:.3f}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "pass_at_k": {"base": base_pk, "mask_lean": mask_lean_pk, "mask_wiki": mask_wiki_pk},
                "verified_disjunction_score": verified_disjunction_score,
                "per_problem": {
                    "base": base_results,
                    "mask_lean": mask_lean_results,
                    "mask_wiki": mask_wiki_results,
                },
            },
            f,
            indent=2,
        )
    print(f"\nFull results saved to: {args.output}")

    summary = build_summary(
        args, len(items), base_pk, mask_lean_pk, mask_wiki_pk,
        drop_lean, drop_wiki, verified_disjunction_score, started_at,
    )
    save_summary(summary, args.summary_output)
    print(f"Summary saved to: {args.summary_output}")


if __name__ == "__main__":
    main()
