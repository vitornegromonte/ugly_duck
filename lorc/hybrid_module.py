from __future__ import annotations

import torch
from torch import Tensor, nn

from .quantization import nf4_quantize, nf4_dequantize


class LoRCLinear(nn.Module):
    def __init__(
        self,
        W_bf16: Tensor,
        V_act: dict[str, Tensor] | None = None,
        U_write: dict[str, Tensor] | None = None,
        quantize_base: bool = True,
        W_q4=None,
    ):
        super().__init__()
        self.in_features = W_bf16.size(1)
        self.out_features = W_bf16.size(0)
        self.base_is_quantized = quantize_base

        if quantize_base:
            q = W_q4 if W_q4 is not None else nf4_quantize(W_bf16)
            if isinstance(q, dict):
                self.register_buffer("_q_packed", q["packed"])
                self.register_buffer("_q_absmax", q["absmax"])
                self._q_group_size = q["group_size"]
                self._q_shape = q["shape"]
            else:
                self.register_buffer("W_base", q)
                self._is_bnb_weight = True
        else:
            self.register_buffer("W_base", W_bf16.contiguous().to(torch.bfloat16))

        self.V_act = nn.ParameterDict()
        self.U_write = nn.ParameterDict()
        if V_act is not None and U_write is not None:
            for domain in V_act:
                v = V_act[domain].contiguous()
                u = U_write[domain].contiguous()
                self.V_act[domain] = nn.Parameter(v, requires_grad=False)
                self.U_write[domain] = nn.Parameter(u, requires_grad=False)

        self.active_profile: str | None = None

    def _dequantized_base(self, dtype: torch.dtype) -> Tensor:
        if not self.base_is_quantized:
            return self.W_base.to(dtype)
        if getattr(self, "_is_bnb_weight", False):
            W = nf4_dequantize(self.W_base).to(dtype)
        else:
            q = {
                "packed": self._q_packed,
                "absmax": self._q_absmax,
                "group_size": self._q_group_size,
                "shape": self._q_shape,
            }
            W = nf4_dequantize(q).to(dtype)
        expected = (self.out_features, self.in_features)
        if tuple(W.shape) != expected:
            raise RuntimeError(
                f"LoRCLinear base dequantized to shape {tuple(W.shape)}, expected {expected}. "
                f"NF4 quantize/dequantize round trip is corrupting shape."
            )
        return W

    def forward(
        self, x: Tensor, profile: str | None = None
    ) -> Tensor:
        W = self._dequantized_base(x.dtype)
        z = x @ W.T

        active = profile if profile is not None else self.active_profile
        if active is not None and active in self.V_act:
            v = self.V_act[active].to(x.dtype)
            u = self.U_write[active].to(x.dtype)
            z = z + (x @ v) @ u.T

        return z

    def correction_overhead_mb(self) -> float:
        total = 0.0
        for domain in self.V_act:
            total += self.V_act[domain].numel() * self.V_act[domain].element_size()
            total += self.U_write[domain].numel() * self.U_write[domain].element_size()
        return total / (1024**2)


def set_profile(model: nn.Module, profile: str | None) -> None:
    for mod in model.modules():
        if isinstance(mod, LoRCLinear):
            mod.active_profile = profile
