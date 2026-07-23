from __future__ import annotations

import torch
from torch import Tensor


def build_correction(
    E: Tensor,
    V_domain: Tensor,
    K: int | None = None,
    module_name: str = "",
) -> tuple[Tensor, Tensor]:
    if K is None:
        K = V_domain.size(-1)
    P = V_domain @ V_domain.T
    E_proj = E.float() @ P.float()
    U, S, Vh = torch.linalg.svd(E_proj, full_matrices=False)
    k = min(K, U.size(0), Vh.size(0))
    U_k, S_k, Vh_k = U[:, :k], S[:k], Vh[:k, :]
    sqrt_S = S_k.clamp(min=1e-10).sqrt()
    V_act = Vh_k.T @ torch.diag(sqrt_S)
    U_write = U_k @ torch.diag(sqrt_S)
    return (V_act.to(torch.bfloat16).contiguous(), U_write.to(torch.bfloat16).contiguous())


def correction_storage_mb(
    modules: dict[str, tuple[Tensor, Tensor]],
) -> float:
    total = 0.0
    for name, (V_act, U_write) in modules.items():
        total += V_act.numel() * V_act.element_size()
        total += U_write.numel() * U_write.element_size()
    return total / (1024**2)
