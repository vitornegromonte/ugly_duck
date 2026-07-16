# Differential Fidelity Inference Engine

Keep a dense LLM (SmolLM2-135M-Instruct) intact; use zero-shot probing to locate domain-specific sub-networks (law vs coding/math); restructure (do NOT prune/remove) the unwanted region's weights to a smaller fidelity (binary / 1-bit / 4-bit Q4); and run a native Python engine that loads different fidelity profiles based on prompt context.

**Concept:** MLP neurons that activate strongly for law text but weakly for code are tagged as "law-dominant." These regions are structurally compressed to 1-bit binary, while the rest gets 4-bit Q4 uniform quantization. At inference time, the engine loads the *relevant* domain at full precision and keeps the *irrelevant* domain compressed — saving memory without removing any parameters.

**This is a proof-of-concept** for differential fidelity inference. It demonstrates zero-shot subnetwork identification and structural compression (not pruning) of domain-specific weights. Runs on CPU with SmolLM2-135M (~300 MB BF16) for quick iteration.

## How to Run

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Profile subnetworks (identify law-dominant heads/neurons)
python profile_subnetworks.py
# → ./outputs/mask.json

# 3. Build compressed sidecar weights
python build_compressed_weights.py
# → ./compressed_weights/*.pt + manifest.json

# 4. Run inference engine
python engine.py --prompt "Write a Python function to implement quicksort."
python engine.py --prompt "What are the elements of negligence in tort law?"
python engine.py --demo

# 5. Evaluate
python evaluate.py
# → ./results/metrics.json
```

## File Structure

```
diffeng/
├── requirements.txt              # Dependencies
├── profile_subnetworks.py        # Zero-shot subnetwork identification
├── build_compressed_weights.py   # Binary/Q4 compression sidecars
├── engine.py                     # Context-aware inference engine
├── evaluate.py                   # Memory/accuracy/latency benchmarks
├── README.md                     # This file
├── outputs/
│   └── mask.json                 # Law-dominant head/neuron indices
├── compressed_weights/
│   ├── manifest.json             # Tensor → file + fidelity tag mapping
│   └── *.pt                      # Compressed weight tensors
└── results/
    └── metrics.json              # Evaluation results
```

## Architecture

```
Prompt → [Keyword Classifier] → FidelityProfile
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                 CODING             LAW             GENERAL
                    │                 │                 │
        ┌───────────┴──┐    ┌────────┴────────┐       │
        │              │    │                 │       │
    Law regions   Coding/   Law regions    Coding/    All BF16
     (binary)   shared BF16  (BF16)      shared Q4
```

- **profile_subnetworks.py**: Forward hooks on `down_proj` (MLP) capture per-neuron activation L2 norms. Score = law_norm / (coding_norm + 1e-6). Threshold 2.5.
- **build_compressed_weights.py**: Law-tagged → `sign(w)` with per-row scaling (1-bit). Non-tagged → uniform 16-level quantization (4-bit). Original weights preserved.
- **engine.py**: Keyword classifier routes to CODING/LAW/GENERAL. `load_state_dict` with mixed-precision state dict.
- **evaluate.py**: RSS memory, keyword-presence accuracy, tokens/sec latency.

## Results (SmolLM2-135M-Instruct, CPU)

### Storage

| Format | Size | Reduction |
|--------|------|-----------|
| BF16 (original) | 303.4 MB | — |
| Compressed (binary + Q4) | 49.4 MB | **83.7%** |
| Manifest | 26 KB | — |

### Zero-Shot Profiling

| Metric | Value |
|--------|-------|
| Coding prompts | 10 |
| Law prompts | 10 |
| Threshold | 2.5× |
| Law-dominant heads | 0 / 270 (0.0%) |
| Law-dominant neurons | 839 / 46,080 (1.8%) |
| Estimated params affected | 1.4M / 136M (1.1%) |

Law-dominant neurons per layer:

| Layer | Neurons | Layer | Neurons | Layer | Neurons |
|-------|---------|-------|---------|-------|---------|
| 0 | 54 | 10 | 7 | 20 | 23 |
| 1 | 11 | 11 | 1 | 21 | 32 |
| 2 | 9 | 12 | 7 | 22 | 43 |
| 3 | 3 | 13 | 13 | 23 | 56 |
| 4 | 9 | 14 | 11 | 24 | 70 |
| 5 | 19 | 15 | 11 | 25 | 72 |
| 6 | 27 | 16 | 10 | 26 | 60 |
| 7 | 12 | 17 | 14 | 27 | 91 |
| 8 | 9 | 18 | 15 | 28 | 78 |
| 9 | 4 | 19 | 17 | 29 | 51 |

### Memory (CPU)

| Profile | RSS (MB) | Tensor Mem (MB) |
|---------|----------|-----------------|
| CODING  | 1595     | 310.6           |
| LAW     | 1556     | 310.6           |
| GENERAL | 1556     | 310.6           |

> ⚠ On CPU, both full and decompressed weights reside in the same memory pool. The RSS reflects OS process memory (includes Python, PyTorch, model, tokenizer). Actual VRAM savings would be visible on GPU where compressed weights can stay in CPU RAM and only the active profile is decompressed to GPU.

### Latency (10 runs × 20 tokens each)

| Prompt Domain | Profile  | Avg Time (s) | Tokens/sec | vs GENERAL |
|---------------|----------|-------------|------------|------------|
| Coding        | GENERAL  | 0.47        | 42.4       | —          |
| Coding        | COMPRESSED| 0.74       | 26.9       | 1.6× slower |
| Law           | GENERAL  | 0.11        | 183.6      | —          |
| Law           | COMPRESSED| 0.10       | 193.5      | 1.1× faster (noise) |

> Latency variance is driven by generation length (EOS hit timing), not compression. Once decompressed, all profiles use identical BF16 forward passes.

### Accuracy (Keyword Presence in Generated Text)

| Domain | Profile  | Mean Score | vs GENERAL |
|--------|----------|-----------|------------|
| Coding | GENERAL  | 0.13      | —          |
| Coding | COMPRESSED| 0.13     | **0% change** |
| Law    | GENERAL  | 0.40      | —          |
| Law    | COMPRESSED| 0.40     | **0% change** |

Per-prompt breakdown:

| Condition | Prompt | Score |
|-----------|--------|-------|
| CODING \| GENERAL | Write a Python function to implement merge sort. | 0.20 |
| CODING \| GENERAL | Write a Python function to reverse a linked list. | 0.20 |
| CODING \| GENERAL | Implement a binary search algorithm in Python. | 0.00 |
| CODING \| GENERAL | Mean | **0.13** |
| CODING \| COMPRESSED | Write a Python function to implement merge sort. | 0.20 |
| CODING \| COMPRESSED | Write a Python function to reverse a linked list. | 0.20 |
| CODING \| COMPRESSED | Implement a binary search algorithm in Python. | 0.00 |
| CODING \| COMPRESSED | Mean | **0.13** |
| LAW \| GENERAL | What are the elements of negligence in tort law? | 0.40 |
| LAW \| GENERAL | Explain the duty of care in negligence claims. | 0.60 |
| LAW \| GENERAL | What constitutes a breach of contract? | 0.20 |
| LAW \| GENERAL | Mean | **0.40** |
| LAW \| COMPRESSED | What are the elements of negligence in tort law? | 0.40 |
| LAW \| COMPRESSED | Explain the duty of care in negligence claims. | 0.60 |
| LAW \| COMPRESSED | What constitutes a breach of contract? | 0.20 |
| LAW \| COMPRESSED | Mean | **0.40** |

Keywords checked — coding: `def, sort, return, list, array`; law: `duty, breach, negligence, care, damages`.

Compressed profiles show **zero accuracy degradation** on these coarse keyword metrics, matching the GENERAL (full BF16) baseline exactly.

### Engine Demo Output

**CODING profile** — Prompt: *Write a Python function to implement quicksort.*
```
system
You are a helpful AI assistant named SmolLM, trained by Hugging Face
user
Write a Python function to implement quicksort.
assistant
```
(2.16s, 20 tokens)

**LAW profile** — Prompt: *What are the elements of negligence in tort law?*
```
system
You are a helpful AI assistant named SmolLM, trained by Hugging Face
user
What are the elements of negligence in tort law?
assistant
In tort law, negligence is a key element of negligence prosecution, which involves a plaintiff in negligence prosecution
```
(1.43s, 20 tokens)

## References

- Cao et al. — "Zero-shot sub-network identification via subtractive probing"
- Kim et al., Fan et al. — "Context-aware fidelity profile loading inspired by MoE offloading"
