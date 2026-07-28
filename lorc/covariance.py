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
) -> dict[tuple[str, str], tuple[Tensor, Tensor, int]]:
    model.eval()
    model.to(device)

    target_modules = _find_target_modules(model, target_patterns)
    pre_cache: dict[tuple[str, str], tuple[Tensor, Tensor, int]] = {}
    post_cache: dict[tuple[str, str], tuple[Tensor, Tensor, int]] = {}

    def _accumulate(cache, key, flat):
        n = flat.size(0)
        xx = (flat.T @ flat).cpu()
        xs = flat.sum(dim=0).cpu()
        if key in cache:
            cum_xx, cum_xs, cum_n = cache[key]
            cache[key] = (cum_xx + xx, cum_xs + xs, cum_n + n)
        else:
            cache[key] = (xx, xs, n)

    def make_hook(module_name: str):
        def hook(_mod, input, output):
            x = input[0].detach().float()
            x_flat = x.view(-1, x.size(-1))
            _accumulate(pre_cache, (module_name, "pre"), x_flat)

            y = output.detach().float()
            if isinstance(y, tuple):
                y = y[0]
            y_flat = y.view(-1, y.size(-1))
            _accumulate(post_cache, (module_name, "post"), y_flat)

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

    merged: dict[tuple[str, str], tuple[Tensor, Tensor, int]] = {}
    for k, v in pre_cache.items():
        merged[k] = v
    for k, v in post_cache.items():
        merged[k] = v
    return merged


def cache_to_covariance(
    cache: dict[tuple[str, str], tuple[Tensor, Tensor, int]]
) -> dict[tuple[str, str], Tensor]:
    result: dict[tuple[str, str], Tensor] = {}
    for k, (cum_xx, cum_xs, n) in cache.items():
        n = max(n, 1)
        mean = cum_xs / n
        result[k] = cum_xx / n - torch.outer(mean, mean)
    return result


def domain_subspaces(
    C_lean: dict[tuple[str, str], Tensor],
    C_wiki: dict[tuple[str, str], Tensor],
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[tuple[str, str], tuple[Tensor, Tensor]]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    subspaces: dict[tuple[str, str], tuple[Tensor, Tensor]] = {}
    for key in C_lean:
        if key not in C_wiki:
            continue
        delta = alpha * C_lean[key] - beta * C_wiki[key]
        delta = (delta + delta.T) / 2
        delta = delta.to(device)
        eigvals, eigvecs = torch.linalg.eigh(delta)
        idx_lean = torch.argsort(eigvals, descending=True)
        V_lean = eigvecs[:, idx_lean[:K]]
        idx_wiki = torch.argsort(eigvals, descending=False)
        V_wiki = eigvecs[:, idx_wiki[:K]]
        subspaces[key] = (V_lean.contiguous(), V_wiki.contiguous())
    return subspaces
