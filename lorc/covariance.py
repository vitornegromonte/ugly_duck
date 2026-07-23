from __future__ import annotations

import re
import torch
from torch import Tensor
from torch import nn
from collections.abc import Iterator


def _target_module_filter(name: str, patterns: list[str]) -> bool:
    return any(re.match(p, name) for p in patterns)


def _find_target_modules(model: nn.Module, patterns: list[str]) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if _target_module_filter(name, patterns):
            modules[name] = mod
    return modules


def collect_covariances(
    model: nn.Module,
    dataloader: Iterator[Tensor],
    target_patterns: list[str],
    n_prompts: int,
    device: str = "cuda",
) -> dict[tuple[str, str], tuple[Tensor, int]]:
    model.eval()
    model.to(device)

    target_modules = _find_target_modules(model, target_patterns)
    pre_cache: dict[tuple[str, str], tuple[Tensor, int]] = {}
    post_cache: dict[tuple[str, str], tuple[Tensor, int]] = {}

    def make_hook(module_name: str):
        def hook(_mod, input, output):
            x = input[0].detach().float()
            x_flat = x.view(-1, x.size(-1))
            n = x_flat.size(0)
            key = (module_name, "pre")
            if key in pre_cache:
                cum_sum, cum_n = pre_cache[key]
                pre_cache[key] = (cum_sum + x_flat.T @ x_flat, cum_n + n)
            else:
                pre_cache[key] = (x_flat.T @ x_flat, n)

            y = output.detach().float()
            if isinstance(y, tuple):
                y = y[0]
            y_flat = y.view(-1, y.size(-1))
            n_y = y_flat.size(0)
            key = (module_name, "post")
            if key in post_cache:
                cum_sum, cum_n = post_cache[key]
                post_cache[key] = (cum_sum + y_flat.T @ y_flat, cum_n + n_y)
            else:
                post_cache[key] = (y_flat.T @ y_flat, n_y)

        return hook

    hooks = []
    for name, mod in target_modules.items():
        hooks.append(mod.register_forward_hook(make_hook(name)))

    seen = 0
    log_interval = max(1, n_prompts // 10)
    for input_ids in dataloader:
        if seen >= n_prompts:
            break
        input_ids = input_ids.to(device)
        with torch.no_grad():
            model(input_ids)
        seen += input_ids.size(0)
        if seen % log_interval < input_ids.size(0) or seen >= n_prompts:
            print(f"    Covariance: {min(seen, n_prompts)}/{n_prompts} prompts")

    for h in hooks:
        h.remove()

    merged: dict[tuple[str, str], tuple[Tensor, int]] = {}
    for k, v in pre_cache.items():
        merged[k] = v
    for k, v in post_cache.items():
        merged[k] = v
    return merged


def cache_to_covariance(
    cache: dict[tuple[str, str], tuple[Tensor, int]]
) -> dict[tuple[str, str], Tensor]:
    return {k: cum_sum / max(n, 1) for k, (cum_sum, n) in cache.items()}


def domain_subspaces(
    C_lean: dict[tuple[str, str], Tensor],
    C_wiki: dict[tuple[str, str], Tensor],
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[tuple[str, str], tuple[Tensor, Tensor]]:
    subspaces: dict[tuple[str, str], tuple[Tensor, Tensor]] = {}
    for key in C_lean:
        if key not in C_wiki:
            continue
        delta = alpha * C_lean[key] - beta * C_wiki[key]
        delta = (delta + delta.T) / 2
        eigvals, eigvecs = torch.linalg.eigh(delta)
        idx_lean = torch.argsort(eigvals, descending=True)
        V_lean = eigvecs[:, idx_lean[:K]]
        idx_wiki = torch.argsort(eigvals, descending=False)
        V_wiki = eigvecs[:, idx_wiki[:K]]
        subspaces[key] = (V_lean.contiguous(), V_wiki.contiguous())
    return subspaces
