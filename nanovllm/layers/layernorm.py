"""
RMSNorm — Root Mean Square Layer Normalization
================================================
与标准 LayerNorm 的区别:
  LayerNorm: y = (x - mean) / std * weight + bias
  RMSNorm:   y = x / RMS(x) * weight

RMSNorm 更快（省去均值计算和偏置），效果相当。

融合残差优化:
  传统写法:
    x = x + residual     # 一次显存读写
    x = rms_norm(x)      # 又一次显存读写
  融合写法:
    x, residual = add_rms_forward(x, residual)
    # 在一个 kernel 里完成 add + norm，减少显存带宽
"""

import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放参数

    @torch.compile  # JIT 编译，与前后算子融合
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准 RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight"""
        orig_dtype = x.dtype
        x = x.float()                            # 提升精度计算
        var = x.pow(2).mean(dim=-1, keepdim=True) # 均方值
        x.mul_(torch.rsqrt(var + self.eps))       # x / sqrt(var + eps)，in-place
        x = x.to(orig_dtype).mul_(self.weight)    # 恢复精度 × 缩放
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合残差 + RMSNorm。

        步骤:
          1. x = x + residual（在 float 精度下）
          2. residual = x（保存为下一次的残差）
          3. y = RMSNorm(x)

        返回: (y, residual)
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())     # x = x + residual
        residual = x.to(orig_dtype)               # 保存新残差
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        如果 residual 为 None → 标准 RMSNorm
        如果 residual 不为 None → 融合残差 + RMSNorm
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
