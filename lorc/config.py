from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LoRCConfig:
    K: int = 32
    n_prompts: int = 5000
    batch_size: int = 8
    seq_len: int = 512
    alpha: float = 1.0
    beta: float = 1.0
    causal_filter_method: Literal["percentile", "relative"] | None = "percentile"
    causal_keep_pct: float = 0.5
    causal_rel_threshold: float = 0.01
    causal_n_steps: int = 10
    target_modules: list[str] = field(default_factory=lambda: [
        r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)"
    ])
    include_attention: bool = False
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: str = "cuda"
    output_dir: str = "./results"
    seed: int = 42
