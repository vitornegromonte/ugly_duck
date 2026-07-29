from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import bitsandbytes as bnb

    has_bitsandbytes = True
except ImportError:
    has_bitsandbytes = False


def nf4_quantize(W_bf16: Tensor, group_size: int = 64) -> tuple:
    if has_bitsandbytes:
        linear = bnb.nn.Linear4bit(
            W_bf16.size(0),
            W_bf16.size(1),
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
        )
        linear.weight.data = W_bf16.contiguous()
        return linear.weight
    return _fallback_quantize(W_bf16, group_size)


def nf4_dequantize(qweight) -> Tensor:
    if has_bitsandbytes and hasattr(qweight, "dequantize"):
        return qweight.dequantize().to(torch.bfloat16)
    return _fallback_dequantize(qweight)


NF4_CODE = torch.tensor([
    -1.0, -0.696, -0.525, -0.395, -0.284, -0.184, -0.091,
    0.0, 0.079, 0.161, 0.251, 0.355, 0.479, 0.639, 0.869, 1.0
])


def _nearest_nf4(x: Tensor) -> Tensor:
    code = NF4_CODE.to(device=x.device, dtype=x.dtype)
    dist = (x.unsqueeze(-1) - code).abs()
    return dist.argmin(dim=-1).to(torch.uint8)


def _fallback_quantize(W: Tensor, group_size: int = 64) -> dict:
    W = W.float()
    d_out, d_in = W.shape
    n_groups = (d_in + group_size - 1) // group_size
    pad = n_groups * group_size - d_in
    W_padded = F.pad(W, (0, pad)) if pad else W
    W_flat = W_padded.view(d_out, n_groups, group_size)
    absmax = W_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scaled = W_flat / absmax
    indices = _nearest_nf4(scaled).reshape(d_out, n_groups * group_size)
    even = indices[:, 0::2]
    odd = indices[:, 1::2]
    packed = (odd << 4) | even
    return {
        "packed": packed.to(torch.uint8),
        "absmax": absmax.squeeze(-1).to(torch.float16),
        "group_size": group_size,
        "shape": (d_out, d_in),
    }


def _fallback_dequantize(q: dict) -> Tensor:
    d_out, d_in = q["shape"]
    gs = q["group_size"]
    packed = q["packed"]
    n_groups = q["absmax"].size(-1)
    odd = (packed >> 4).to(torch.long)
    even = (packed & 0x0F).to(torch.long)
    n_even = even.size(1)
    indices = torch.zeros(d_out, 2 * n_even, dtype=torch.long, device=packed.device)
    indices[:, 0::2] = even
    indices[:, 1::2] = odd
    row_vals = NF4_CODE.to(packed.device)[indices].reshape(d_out, n_groups, gs)
    absmax = q["absmax"].to(row_vals.device).unsqueeze(-1)
    out = (row_vals * absmax).reshape(d_out, n_groups * gs)
    return out[:, :d_in].to(torch.bfloat16)
