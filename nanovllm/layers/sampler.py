"""
Sampler — 采样器
=================
根据 logits 和 temperature 采样下一个 token。

使用 Gumbel-Max Trick 实现高效采样:
  sample_token = argmax(logits / temperature + Gumbel(0,1))
  等价于从 softmax(logits/temperature) 定义的多项分布中采样。

Gumbel 噪声: -log(-log(U)) where U ~ Uniform(0,1)
代码中的实现: exponential_(1).clamp_min_(1e-10)
  指数分布 Exp(1) 的采样 = -log(U)，等效于 Gumbel 噪声的一个分量。
  加上 clamp 防止数值问题（log(0) → -inf）。
"""

import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        从 logits 中采样。

        参数:
          logits:       [batch, vocab_size]  输出 logits
          temperatures: [batch]              每个序列的采样温度

        返回:
          token_ids: [batch] 采样出的 token ID

        采样过程:
          1. logits / temperature + Gumbel 噪声
          2. argmax（等价于从 softmax 分布中随机采样）
        """
        # 除以温度
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 计算 softmax 概率
        probs = torch.softmax(logits, dim=-1)

        # Gumbel-Max Trick: argmax(probs / Exponential(1))
        # Exponential(1) = -log(Uniform(0,1))
        # probs / Exp(1) 相当于 probs * Exp(1) 的倒数... 不完全是
        # 实际上是 probs.div_(Exp(1)) → 然后 argmax
        # 这等价于从 categorical 分布中采样
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return sample_tokens
