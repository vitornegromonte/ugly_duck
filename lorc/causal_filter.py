from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
from collections.abc import Iterator


def causal_filter(
    W_q4_weight: Tensor,
    V_act: Tensor,
    U_write: Tensor,
    model: nn.Module,
    module_path: str,
    dataloader: Iterator[tuple[Tensor, Tensor]],
    n_steps: int = 10,
    method: str = "percentile",
    keep_pct: float = 0.5,
    rel_threshold: float = 0.01,
    device: str = "cuda",
) -> tuple[Tensor, Tensor]:
    K = V_act.size(-1)
    model.eval()
    model.to(device)

    mask = nn.Parameter(torch.ones(K, device=device))
    opt = torch.optim.Adam([mask], lr=0.1)

    target_mod = model.get_submodule(module_path)
    d_out, d_in = target_mod.weight.shape
    V_act = V_act.to(device)
    U_write = U_write.to(device)
    orig_weight = target_mod.weight.data.clone()

    def make_hook():
        def hook(mod, input, output):
            with torch.enable_grad():
                x = input[0].float()
                m = torch.sigmoid(mask)
                correction = (x @ V_act) @ (U_write * m.unsqueeze(0)).T
            return (output.float() + correction).to(output.dtype)
        return hook

    handle = target_mod.register_forward_hook(make_hook())

    for step in range(n_steps):
        for input_ids, domain_labels in dataloader:
            input_ids = input_ids.to(device)
            lean_mask = domain_labels > 0.5
            if not lean_mask.any():
                continue

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                reduction="mean",
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            break

    handle.remove()

    with torch.no_grad():
        final_mask = torch.sigmoid(mask)

    if method == "percentile":
        thr = torch.quantile(final_mask, 1.0 - keep_pct)
        keep = final_mask > thr
    elif method == "relative":
        max_grad = final_mask.abs().max()
        keep = final_mask > rel_threshold * max_grad
    else:
        keep = torch.ones(K, dtype=torch.bool, device=device)

    return V_act[:, keep].contiguous(), U_write[keep, :].contiguous()
