"""
RotaryEmbedding — RoPE 旋转位置编码
=====================================
RoPE (Rotary Position Embedding) 通过旋转变换将位置信息注入 Q 和 K:

  Q_rope = Q * cos(pos) + rotate_half(Q) * sin(pos)
  K_rope = K * cos(pos) + rotate_half(K) * sin(pos)

其中:
  rotate_half([x0, x1, x2, x3, ...]) = [-x1, x0, -x3, x2, ...]
  freqs_i = base^(-2i/d), i = 0, 1, ..., d/2-1
  cos = cos(pos * freqs), sin = sin(pos * freqs)

优点:
  - 位置信息通过旋转注入，不增加额外参数
  - 自然支持外推（通过调整 base 频率）
  - 相对位置信息隐含在 QK 内积中
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    对输入 x 应用旋转变换。

    x 沿最后一维对半分: [x1, x2]
    旋转: [x1*cos - x2*sin, x2*cos + x1*sin]

    这等价于复数乘法: (x1 + i*x2) * (cos + i*sin)
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size  # 当前实现要求对所有维度做旋转

        # 计算频率: inv_freq_i = 1 / base^(2i/d), i=0,1,...,d/2-1
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        # 所有可能位置的频率外积: freqs[pos][i] = pos * inv_freq_i
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # 预计算 cos 和 sin: [max_position, rotary_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()

        # 合并为 [max_position, 1, rotary_dim]，方便广播
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,   # [num_tokens] 位置索引
        query: torch.Tensor,       # [num_tokens, num_heads, head_dim]
        key: torch.Tensor,         # [num_tokens, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从预计算的 cache 中取对应位置的 cos/sin，应用到 Q 和 K"""
        cos_sin = self.cos_sin_cache[positions]  # [num_tokens, 1, head_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)       # 前半是 cos，后半是 sin
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)  # 缓存一个实例（相同参数只需创建一次）
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """工厂函数，带 LRU 缓存"""
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
