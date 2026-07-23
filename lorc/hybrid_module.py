from __future__ import annotations

import torch
from torch import Tensor, nn

from .quantization import nf4_quantize, has_bitsandbytes


class LoRCLinear(nn.Module):
    def __init__(
        self,
        W_bf16: Tensor,
        V_act: dict[str, Tensor] | None = None,
        U_write: dict[str, Tensor] | None = None,
        quantize_base: bool = True,
    ):
        super().__init__()
        self.in_features = W_bf16.size(1)
        self.out_features = W_bf16.size(0)

        if quantize_base and has_bitsandbytes:
            self.register_buffer("W_base", nf4_quantize(W_bf16))
            self.base_is_quantized = True
        else:
            self.register_buffer("W_base", W_bf16.contiguous().to(torch.bfloat16))
            self.base_is_quantized = False

        self.V_act = nn.ParameterDict()
        self.U_write = nn.ParameterDict()
        if V_act is not None and U_write is not None:
            for domain in V_act:
                v = V_act[domain].contiguous()
                u = U_write[domain].contiguous()
                self.V_act[domain] = nn.Parameter(v, requires_grad=False)
                self.U_write[domain] = nn.Parameter(u, requires_grad=False)

    def forward(
        self, x: Tensor, profile: str | None = None
    ) -> Tensor:
        if self.base_is_quantized:
            W = self.W_base.dequantize().to(x.dtype)
        else:
            W = self.W_base.to(x.dtype)
        z = x @ W.T

        if profile is not None and profile in self.V_act:
            z = z + (x @ self.V_act[profile]) @ self.U_write[profile].T

        return z

    def correction_overhead_mb(self) -> float:
        total = 0.0
        for domain in self.V_act:
            total += self.V_act[domain].numel() * self.V_act[domain].element_size()
            total += self.U_write[domain].numel() * self.U_write[domain].element_size()
        return total / (1024**2)
