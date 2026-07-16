#!/usr/bin/env python3
"""
evaluate.py — Benchmark the differential fidelity inference engine.

Tests:
  1. Memory footprint per profile (RSS + tensor memory)
  2. Accuracy/perplexity sanity check (keyword-presence in generated text)
  3. Latency (tokens/sec for 10 forward passes per profile)

Output: stdout markdown tables + ./results/metrics.json
"""

import json
import os
import time
import psutil
import torch
from engine import DifferentialEngine, FidelityProfile, MODEL_NAME

OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CODING_PROMPTS_EVAL = [
    "Write a Python function to implement merge sort.",
    "Write a Python function to reverse a linked list.",
    "Implement a binary search algorithm in Python.",
]

LAW_PROMPTS_EVAL = [
    "What are the elements of negligence in tort law?",
    "Explain the duty of care in negligence claims.",
    "What constitutes a breach of contract?",
]

CODING_KEYWORDS = ["def", "sort", "return", "list", "array"]
LAW_KEYWORDS = ["duty", "breach", "negligence", "care", "damages"]


def measure_memory(engine: DifferentialEngine):
    results = {}
    process = psutil.Process()
    for profile in FidelityProfile:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        engine.load_profile(profile)
        _ = engine.generate("hello world", max_new_tokens=1)
        rss_mb = process.memory_info().rss / (1024 * 1024)
        tensor_mem = sum(
            v.numel() * v.element_size()
            for v in engine.model.state_dict().values()
        ) / (1024 * 1024)
        results[profile.value] = {"rss_mb": rss_mb, "tensor_mem_mb": tensor_mem}
    return results


def accuracy_check(engine: DifferentialEngine):
    results = {}

    # Coding prompts: compare CODING profile vs GENERAL baseline
    for profile_label, profile in [("coding_compressed", FidelityProfile.CODING),
                                    ("coding_general", FidelityProfile.GENERAL)]:
        engine.load_profile(profile)
        scores = []
        for prompt in CODING_PROMPTS_EVAL:
            output = engine.generate(prompt, max_new_tokens=30)
            generated = output.lower()
            hits = sum(1 for kw in CODING_KEYWORDS if kw in generated)
            scores.append(hits / len(CODING_KEYWORDS))
        results[profile_label] = {
            "scores": scores,
            "mean": sum(scores) / len(scores),
        }

    # Law prompts: compare LAW profile vs GENERAL baseline
    for profile_label, profile in [("law_compressed", FidelityProfile.LAW),
                                    ("law_general", FidelityProfile.GENERAL)]:
        engine.load_profile(profile)
        scores = []
        for prompt in LAW_PROMPTS_EVAL:
            output = engine.generate(prompt, max_new_tokens=30)
            generated = output.lower()
            hits = sum(1 for kw in LAW_KEYWORDS if kw in generated)
            scores.append(hits / len(LAW_KEYWORDS))
        results[profile_label] = {
            "scores": scores,
            "mean": sum(scores) / len(scores),
        }

    return results


def latency_test(engine: DifferentialEngine, n_runs: int = 10):
    results = {}
    for profile, prompts in [
        (FidelityProfile.CODING, CODING_PROMPTS_EVAL),
        (FidelityProfile.LAW, LAW_PROMPTS_EVAL),
    ]:
        engine.load_profile(profile)
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
        results[profile.value] = {
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

    print("[1/3] Memory footprint test...")
    mem_results = measure_memory(engine)

    print("[2/3] Accuracy/perplexity sanity check...")
    acc_results = accuracy_check(engine)

    print("[3/3] Latency test...")
    lat_results = latency_test(engine, n_runs=10)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n### Memory Footprint")
    print("| Profile    | RSS (MB) | Tensor Mem (MB) |")
    print("|------------|----------|-----------------|")
    for p, d in mem_results.items():
        print(f"| {p:<10} | {d['rss_mb']:>8.0f} | {d['tensor_mem_mb']:>15.1f} |")

    print("\n### Accuracy (Keyword Presence)")
    print("| Profile  | Prompt | Score |")
    print("|----------|--------|-------|")
    for p, d in acc_results.items():
        prompts = CODING_PROMPTS_EVAL if "coding" in p else LAW_PROMPTS_EVAL
        for i, score in enumerate(d["scores"]):
            print(f"| {p:<9} | {prompts[i][:40]:40} | {score:.2f} |")
        print(f"| {p:<9} | {'Mean':40} | {d['mean']:.2f} |")

    print("\n### Latency")
    print("| Profile  | Avg Time (s) | Tokens/sec |")
    print("|----------|-------------|------------|")
    for p, d in lat_results.items():
        print(f"| {p:<9} | {d['avg_time_s']:>12.3f} | {d['tokens_per_sec']:>10.1f} |")

    metrics = {"memory": mem_results, "accuracy": acc_results, "latency": lat_results}
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
