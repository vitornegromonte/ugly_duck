#!/usr/bin/env python3
"""
evaluate.py — Benchmark the differential fidelity inference engine.

Tests:
  1. Memory footprint per profile (RSS + tensor memory)
  2. Perplexity on held-out domain text per profile + cross-domain
  3. Keyword-presence accuracy (as secondary metric)
  4. Latency (tokens/sec)

Output: stdout markdown tables + ./results/metrics.json
"""

import json
import math
import os
import time
import psutil
import torch
from engine import DifferentialEngine, FidelityProfile, MODEL_NAME

OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load held-out prompts
HELD_OUT_PATH = "./outputs/held_out_prompts.json"
with open(HELD_OUT_PATH) as f:
    ho = json.load(f)
CODING_HELD = ho["coding"]
LAW_HELD = ho["law"]

CODING_KEYWORDS = ["def", "sort", "return", "list", "array"]
LAW_KEYWORDS = ["duty", "breach", "negligence", "care", "damages"]


def compute_perplexity(model, tokenizer, texts):
    """Compute mean perplexity over a list of texts."""
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        nll = outputs.loss.item() * inputs["input_ids"].size(1)
        total_nll += nll
        total_tokens += inputs["input_ids"].size(1)
    return math.exp(total_nll / total_tokens)


def measure_memory(engine: DifferentialEngine):
    results = {}
    process = psutil.Process()
    for profile in FidelityProfile:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        engine.load_profile(profile)
        _ = engine.generate("hello", max_new_tokens=1)
        rss_mb = process.memory_info().rss / (1024 * 1024)
        tensor_mem = sum(
            v.numel() * v.element_size()
            for v in engine.model.state_dict().values()
        ) / (1024 * 1024)
        results[profile.value] = {"rss_mb": rss_mb, "tensor_mem_mb": tensor_mem}
    return results


def accuracy_check(engine: DifferentialEngine):
    results = {}

    for profile, label, prompts, keywords in [
        (FidelityProfile.CODING, "coding", CODING_HELD, CODING_KEYWORDS),
        (FidelityProfile.GENERAL, "coding_general", CODING_HELD, CODING_KEYWORDS),
        (FidelityProfile.LAW, "law", LAW_HELD, LAW_KEYWORDS),
        (FidelityProfile.GENERAL, "law_general", LAW_HELD, LAW_KEYWORDS),
    ]:
        engine.load_profile(profile)
        scores = []
        for prompt in prompts:
            output = engine.generate(prompt, max_new_tokens=30)
            generated = output.lower()
            hits = sum(1 for kw in keywords if kw in generated)
            scores.append(hits / len(keywords))
        results[label] = {
            "scores": scores,
            "mean": sum(scores) / len(scores),
        }

    return results


def latency_test(engine: DifferentialEngine, n_runs: int = 10):
    results = {}

    for profile, base_label in [
        (FidelityProfile.CODING, "coding"),
        (FidelityProfile.GENERAL, "coding_general"),
        (FidelityProfile.LAW, "law"),
        (FidelityProfile.GENERAL, "law_general"),
    ]:
        engine.load_profile(profile)
        prompts = CODING_HELD if "coding" in base_label else LAW_HELD
        times = []
        total_tokens = 0
        for i in range(n_runs):
            prompt = prompts[i % len(prompts)]
            inputs = engine.tokenizer(prompt, return_tensors="pt").to("cpu")
            t0 = time.time()
            with torch.no_grad():
                _ = engine.model.generate(
                    **inputs, max_new_tokens=20,
                    pad_token_id=engine.tokenizer.pad_token_id,
                )
            elapsed = time.time() - t0
            times.append(elapsed)
            total_tokens += 20
        results[base_label] = {
            "avg_time_s": sum(times) / len(times),
            "tokens_per_sec": total_tokens / sum(times),
        }

    return results


def main():
    print("=" * 60)
    print("Differential Fidelity — Evaluation Suite")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\nInitializing engine...")
    engine = DifferentialEngine()

    print("[1/4] Memory footprint test...")
    mem_results = measure_memory(engine)

    print("[2/4] Perplexity test...")
    ppl_results = {}
    for profile, label, eval_prompts in [
        (FidelityProfile.CODING, "coding_on_coding", CODING_HELD),
        (FidelityProfile.LAW, "law_on_coding", CODING_HELD),
        (FidelityProfile.GENERAL, "general_on_coding", CODING_HELD),
        (FidelityProfile.LAW, "law_on_law", LAW_HELD),
        (FidelityProfile.CODING, "coding_on_law", LAW_HELD),
        (FidelityProfile.GENERAL, "general_on_law", LAW_HELD),
    ]:
        engine.load_profile(profile)
        ppl = compute_perplexity(engine.model, engine.tokenizer, eval_prompts)
        ppl_results[label] = ppl
        print(f"  {label:<25} ppl={ppl:.2f}")

    print("[3/4] Accuracy check...")
    acc_results = accuracy_check(engine)

    print("[4/4] Latency test...")
    lat_results = latency_test(engine, n_runs=10)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n### Memory")
    print("| Profile    | RSS (MB) | Tensor Mem (MB) |")
    print("|------------|----------|-----------------|")
    for p, d in mem_results.items():
        print(f"| {p:<10} | {d['rss_mb']:>8.0f} | {d['tensor_mem_mb']:>15.1f} |")

    print("\n### Perplexity (held-out, lower = better)")
    print("| Profile      | Coding PPL | Law PPL |")
    print("|-------------|-----------|---------|")
    for prof, c_key, l_key in [
        ("CODING", "coding_on_coding", "coding_on_law"),
        ("LAW", "law_on_coding", "law_on_law"),
        ("GENERAL", "general_on_coding", "general_on_law"),
    ]:
        print(f"| {prof:<12} | {ppl_results[c_key]:>9.2f} | {ppl_results[l_key]:>7.2f} |")

    print("\n### Cross-domain degradation")
    print("| Direction | PPL Increase |")
    print("|-----------|-------------|")
    for profile, match_key, mismatch_key, label in [
        ("CODING", "coding_on_coding", "law_on_coding", "Coding domain under CODING profile"),
        ("CODING", "law_on_law", "coding_on_law", "Law domain under CODING profile"),
        ("LAW", "law_on_law", "coding_on_law", "Law domain under LAW profile"),
        ("LAW", "coding_on_coding", "law_on_coding", "Coding domain under LAW profile"),
    ]:
        ref = ppl_results[match_key]
        test = ppl_results[mismatch_key]
        change = ((test - ref) / ref) * 100
        print(f"| {label:<35} | {change:>+7.1f}% |")

    print("\n### Keyword Accuracy")
    print("| Label          | Mean Score |")
    print("|----------------|-----------|")
    for label, d in sorted(acc_results.items()):
        print(f"| {label:<15} | {d['mean']:>9.2f} |")

    print("\n### Latency")
    print("| Label              | Avg Time (s) | Tokens/sec |")
    print("|--------------------|-------------|------------|")
    for label, d in sorted(lat_results.items()):
        print(f"| {label:<19} | {d['avg_time_s']:>12.3f} | {d['tokens_per_sec']:>10.1f} |")

    metrics = {
        "memory": mem_results,
        "perplexity": ppl_results,
        "accuracy": acc_results,
        "latency": lat_results,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
