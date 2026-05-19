"""
Activation — SiLU 门控激活函数
===============================
SwiGLU 激活: output = SiLU(gate) * up

其中: gate = x[:, :intermediate], up = x[:, intermediate:]
      SiLU(x) = x * sigmoid(x)
"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    将输入沿最后一维对半切分:
      前半 → SiLU 激活
      后半 → 保持原值
      结果 → 逐元素相乘

    使用 torch.compile 做 JIT 编译，与后续的 down_proj 融合。
    """

    @torch.compile  # JIT 编译，减少 Python overhead
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)        # 沿最后一维对半分: x=gate, y=up
        return F.silu(x) * y         # SiLU(gate) * up
