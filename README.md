# Differential Fidelity Inference Engine

Research into **selective degradation of domain-specific sub-networks** in LLMs — keeping a model at near-full precision on its target domain while aggressively compressing weights for irrelevant domains. A practical testbed for the hypothesis that reasoning and general text processing occupy (disjoint or polysemantic) low-rank subspaces.

---

## Projects

### `lorc/` — Low-rank Quantization Correction (active)

Uses **activation covariances** (E[xx^T] on domain-specific inputs) to identify subspaces where reasoning vs general-text signals differ, then projects quantization error onto those subspaces as low-rank BF16 correction factors.

**Core idea:** Keep the full model at NF4 (0.5 bytes/param). Add back a tiny low-rank correction `(x @ V_act) @ U_write^T` in BF16 for the active domain. No decompression of the base weights at inference time.

```
z = quantized_4bit_matmul(x, W_base)     # 68 MB for 135M params
z += (x @ V_act_domain) @ U_write_domain^T  # ~25 MB per domain
```

| Component | Storage | Format |
|-----------|---------|--------|
| W_base (all weights) | ~68 MB | NF4 (per-group) |
| Domain corrections (coding + law) | ~49 MB | BF16 low-rank |
| **Total** | **~117 MB** | vs 272 MB BF16 → **57% savings** |

**Key files:**

| File | Role |
|------|------|
| `covariance.py` | ΔC = α·C_lean − β·C_wiki, eigendecomposition → domain subspaces (pre + post MLP) |
| `correction.py` | Project quantization error onto subspace → SVD → V_act, U_write factors |
| `quantization.py` | bitsandbytes NF4 wrapper + CPU fallback (3.8×, 9.3% error) |
| `causal_filter.py` | Single gradient pass: learn a mask over K components, drop |∇|≈0 |
| `hybrid_module.py` | `LoRCLinear` — NF4 base + `nn.ParameterDict` of domain corrections |
| `ablation.py` | Disjunction score D = ∆PPL_lean / ∆PPL_wiki, subspace cosine similarity |
| `run_pipeline.py` | End-to-end: data → cov → quantize → build → filter → ablate → report |
| `run_sweep.py` | Cartesian hyperparameter sweep (1248 combos via YAML config) |
| `sweep_config.yaml` | Search space: K ∈ {4,8,16,32,64}, α/β ∈ {0.5,1.0,2.0}, 3 filter methods, attention on/off |

**Usage:**

```bash
# Single run
python -m lorc.run_pipeline --K 32

# Full hyperparameter sweep
./sweep.sh --parallel 4

# Sweep on a different model
./sweep.sh --model "meta-llama/Llama-3.2-1B" --parallel 2

# Compare with VPD decomposition (trained separately)
python -m lorc.run_vpd_comparison --vpd-path /path/to/checkpoint.pt
```

**Conditional experiment arms** (gated by flags):

| Flag | Default | Tests |
|------|---------|-------|
| `--attention` | False | LoRC on Q/K/V/O layers too |
| `--causal-filter` | percentile | {percentile, relative, none} |
| `--causal-keep-pct` | 0.5 | What fraction of components survive filtering |
| `--alpha --beta` | 1.0, 1.0 | Reweighting ΔC = α·C_lean − β·C_wiki |
| `--K` | 32 | Component rank per tensor |

---

### `engine.py`, `profile_subnetworks.py` — Zero-shot Cohen's d profiling (reference)

The first iteration: identify domain-specific heads/neurons via Cohen's d of activations, replace with SVD (k=16) or binary (1-bit) compression, switch profiles with an embedding-based router.

**Key results** (SmolLM2-135M-Instruct, all-CPU demo):

| Profile | Coding PPL | Law PPL |
|---------|-----------|---------|
| CODING | 47.80 | 64.17 |
| LAW | 59.63 | 68.66 |
| GENERAL | 47.80 | 64.17 |

Demo outputs after the int8→4-bit fix:

```
CODING:  Here is a Python function that implements quicksort: def qu
LAW:     In tort law, negligence is a type of negligence that occurs when...
```

---

## Research Questions

1. **Are reasoning circuits spatially disjoint from general text circuits?**
   - *Method:* Feed miniF2F (Lean) + Wikipedia through the model, compute ΔC, build V_act/V_wiki subspaces, ablate Lean-dominant components, measure PPL divergence.
   - *Disjoint hypothesis:* Coding PPL rises, Wiki PPL flat → D ≫ 1
   - *Polysemantic hypothesis:* Both rise proportionally → D ≈ 1

2. **Does RL training (instruct vs base) create new reasoning sub-networks?**
   - *Method:* Run LoRC on both SmolLM2-135M (base) and SmolLM2-135M-Instruct; compare subspace alignment and disjunction scores.

3. **Is covariance-based subspace identification competitive with gradient-based methods (VPD)?**
   - *Method:* Measure cosine similarity between V_act (covariance) and VPD read vectors; compare ablation effect sizes.

4. **What's the Pareto frontier of (K, memory savings, PPL degradation, disjunction score)?**
   - *Method:* Full sweep over K ∈ {4,8,16,32,64}, α/β, causal filter methods.

---

## Repository Structure

```
diffeng/
├── lorc/                         # LoRC — main research package
│   ├── config.py                 # Experiment configuration dataclass
│   ├── data.py                   # miniF2F + Wikipedia loaders
│   ├── quantization.py           # bitsandbytes NF4 + CPU fallback
│   ├── covariance.py             # ΔC eigendecomposition (pre + post MLP)
│   ├── correction.py             # Error projection → SVD factors
│   ├── causal_filter.py          # Gradient-based causal trimming
│   ├── hybrid_module.py          # LoRCLinear (NF4 base + BF16 correction)
│   ├── ablation.py               # Disjunction score, perplexity, overlap
│   ├── run_pipeline.py           # End-to-end orchestrator
│   ├── run_sweep.py              # Hyperparameter sweep engine
│   ├── run_vpd_comparison.py     # LoRC vs VPD subspace comparison
│   └── sweep_config.yaml         # Search space definition
├── engine.py                     # Reference: zero-shot profiling + weight patching
├── profile_subnetworks.py        # Cohen's d profiling
├── build_compressed_weights.py   # Per-unit SVD/binary compression
├── evaluate.py                   # Perplexity, cross-domain, memory benchmarks
├── sweep.sh                      # Sweep launcher
├── requirements.txt
├── outputs/                      # Profiling results (mask.json, prototypes, held-out prompts)
├── compressed_weights/           # Sidecar files (manifest.json + .pt)
└── results/                      # Metrics + sweep results
```

## Dependencies

```
torch
transformers
datasets
bitsandbytes          # GPU: NF4 quantization
pyyaml                # Sweep config parsing
# CPU-only fallback works without bitsandbytes
```

## References

- Goodfire AI (2026) — "Virtual Parameter Directions: Interpreting LM Parameters"
- Goodfire AI (2025) — "Stochastic Parameter Decomposition" (arxiv.org/abs/2506.20790)
- Dettmers et al. (2023) — "QLoRA: Efficient Finetuning of Quantized Language Models" (NF4)
- miniF2F dataset — OpenAI, math competition problems for formal verification
