import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor, # target
        k: torch.Tensor, # source
        v: torch.Tensor, # source
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        scaler = np.sqrt(self.d_k)
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scaler
        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.masked_fill(
                mask, -torch.finfo(attention_weight.dtype).max
            )
        attention_weight = nn.functional.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, v)