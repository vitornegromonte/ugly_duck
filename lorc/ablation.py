from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from collections.abc import Callable
from contextlib import contextmanager


def compute_perplexity(
    model: nn.Module,
    dataloader,
    n_batches: int = 10,
    device: str = "cuda",
    pad_token_id: int | None = None,
) -> float:
    model.eval()
    model.to(device)
    total_nll = 0.0
    total_tokens = 0.0
    seen = 0
    for input_ids in dataloader:
        if seen >= n_batches:
            break
        input_ids = input_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids).logits
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = input_ids[:, 1:].reshape(-1)
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none")
        if pad_token_id is not None:
            valid = (shift_labels != pad_token_id).float()
            total_nll += (loss * valid).sum().item()
            total_tokens += valid.sum().item()
        else:
            total_nll += loss.sum().item()
            total_tokens += shift_labels.numel()
        seen += 1
    return math.exp(total_nll / max(total_tokens, 1))


def disjunction_score(
    base_own_ppl: float,
    base_other_ppl: float,
    ablated_own_ppl: float,
    ablated_other_ppl: float,
) -> float:
    delta_own = max(ablated_own_ppl - base_own_ppl, 1e-8)
    delta_other = max(ablated_other_ppl - base_other_ppl, 1e-8)
    return delta_own / delta_other


def project_out_subspace(W: Tensor, V: Tensor, loc: str) -> Tensor:
    V = V.to(W.dtype)
    P = V @ V.T
    eye = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
    if loc == "pre":
        return W @ (eye - P)
    if loc == "post":
        return (eye - P) @ W
    raise ValueError(f"unknown loc: {loc!r} (expected 'pre' or 'post')")


@contextmanager
def ablated_subspace(
    model: nn.Module,
    subspaces: dict[tuple[str, str], tuple[Tensor, Tensor]],
    domain: str,
):
    """Temporarily project the given domain's subspace directions out of every
    targeted weight matrix, then restore the original weights on exit."""
    domain_idx = 0 if domain == "lean" else 1
    sd = model.state_dict()

    grouped: dict[str, list[tuple[str, Tensor]]] = {}
    for (module_name, loc), pair in subspaces.items():
        w_key = module_name + ".weight"
        if w_key not in sd:
            continue
        grouped.setdefault(w_key, []).append((loc, pair[domain_idx]))

    backup = {w_key: sd[w_key].clone() for w_key in grouped}
    try:
        for w_key, ops in grouped.items():
            W = backup[w_key].float()
            for loc, V in ops:
                W = project_out_subspace(W, V.to(W.device).float(), loc)
            sd[w_key].copy_(W.to(sd[w_key].dtype))
        yield
    finally:
        for w_key, orig in backup.items():
            sd[w_key].copy_(orig)


def ablate_and_measure(
    model: nn.Module,
    subspaces: dict[tuple[str, str], tuple[Tensor, Tensor]],
    domain: str,
    dl_lean,
    dl_wiki,
    n_batches: int = 5,
    device: str = "cuda",
    pad_token_id: int | None = None,
) -> tuple[float, float]:
    with ablated_subspace(model, subspaces, domain):
        ppl_lean = compute_perplexity(
            model, dl_lean, n_batches=n_batches, device=device, pad_token_id=pad_token_id
        )
        ppl_wiki = compute_perplexity(
            model, dl_wiki, n_batches=n_batches, device=device, pad_token_id=pad_token_id
        )
    return ppl_lean, ppl_wiki


def component_overlap(
    V_cov: Tensor,
    V_vpd: Tensor,
) -> Tensor:
    V_cov = V_cov / V_cov.norm(dim=0, keepdim=True).clamp(min=1e-8)
    V_vpd = V_vpd / V_vpd.norm(dim=0, keepdim=True).clamp(min=1e-8)
    sim = V_cov.T @ V_vpd
    return sim
