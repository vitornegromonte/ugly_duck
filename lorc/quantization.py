from __future__ import annotations

import torch
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
    dist = (x.unsqueeze(-1) - NF4_CODE).abs()
    return dist.argmin(dim=-1).to(torch.uint8)


def _fallback_quantize(W: Tensor, group_size: int = 64) -> dict:
    W = W.float()
    d_out, d_in = W.shape
    n_groups = (d_in + group_size - 1) // group_size
    W_flat = W.view(d_out, n_groups, group_size)
    absmax = W_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scaled = W_flat / absmax
    indices = _nearest_nf4(scaled)
    packed_list = []
    for r in range(d_out):
        row = indices[r].flatten()
        even = row[0::2]
        odd = row[1::2]
        if odd.size(0) < even.size(0):
            odd = torch.cat([odd, odd.new_zeros(1)])
        packed_list.append((odd << 4) | even)
    return {
        "packed": torch.stack(packed_list).to(torch.uint8),
        "absmax": absmax.squeeze(-1).to(torch.float16),
        "group_size": group_size,
        "shape": W.shape,
    }


def _fallback_dequantize(q: dict) -> Tensor:
    d_out, d_in = q["shape"]
    gs = q["group_size"]
    packed = q["packed"]
    odd = (packed >> 4).to(torch.long)
    even = (packed & 0x0F).to(torch.long)
    rows = []
    for r in range(d_out):
        n_even = even.size(1)
        indices = torch.zeros(2 * n_even, dtype=torch.long, device=packed.device)
        indices[0::2] = even[r]
        indices[1::2] = odd[r]
        row_vals = NF4_CODE.to(packed.device)[indices[:d_in]]
        row_vals = row_vals.reshape(1, d_in // gs, gs)
        absmax_r = q["absmax"][r].unsqueeze(-1)
        rows.append((row_vals * absmax_r).reshape(1, d_in))
    return torch.cat(rows, dim=0).to(torch.bfloat16)
