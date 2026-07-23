from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from collections.abc import Callable


def compute_perplexity(
    model: nn.Module, dataloader, n_batches: int = 10, device: str = "cuda"
) -> float:
    model.eval()
    model.to(device)
    total_nll = 0.0
    total_tokens = 0
    seen = 0
    for input_ids, _ in dataloader:
        if seen >= n_batches:
            break
        input_ids = input_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += input_ids.numel()
        seen += 1
    return math.exp(total_nll / max(total_tokens, 1))


def disjunction_score(
    base_ppl: float,
    ablated_ppl_lean: float,
    ablated_ppl_wiki: float,
) -> float:
    delta_lean = max(ablated_ppl_lean - base_ppl, 1e-8)
    delta_wiki = max(ablated_ppl_wiki - base_ppl, 1e-8)
    return delta_lean / delta_wiki


def component_overlap(
    V_cov: Tensor,
    V_vpd: Tensor,
) -> Tensor:
    V_cov = V_cov / V_cov.norm(dim=0, keepdim=True).clamp(min=1e-8)
    V_vpd = V_vpd / V_vpd.norm(dim=0, keepdim=True).clamp(min=1e-8)
    sim = V_cov.T @ V_vpd
    return sim
