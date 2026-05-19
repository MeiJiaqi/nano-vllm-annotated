"""
Embedding & LM Head — 支持 Tensor Parallelism
===============================================
VocabParallelEmbedding:
  词表按 TP rank 切分，每个 GPU 只存 vocab_size/tp_size 行。
  forward: 对输入 token_id 做 mask（过滤不在本 rank 的 token），
           各自查表后 all-reduce 求和得到完整 embedding。

ParallelLMHead:
  继承 VocabParallelEmbedding，将 hidden_states 投影回词表大小。
  prefill 时只取每个序列最后一个位置的 hidden_states（因为只需预测下一个 token）。
  各 GPU 计算部分 logits，gather 拼接得到完整 logits（与 embedding 不同，
  这里不做 all-reduce 而是拼接，因为 logits 需要完整的词表维度做采样）。
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词嵌入层，词表按 TP rank 切分。

    例: vocab_size=32000, tp_size=4
      rank 0: token 0~7999
      rank 1: token 8000~15999
      rank 2: token 16000~23999
      rank 3: token 24000~32000

    forward 时:
      每个 token_id 只在它所属的 rank 上产生 embedding，
      其他 rank 输出 0，然后 all-reduce 求和。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0

        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 当前 rank 负责的词表范围: [vocab_start_idx, vocab_end_idx)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 只存本 rank 负责的 embedding 行
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """按词表行切分加载"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        TP Embedding 前向。

        输入: x [num_tokens] — token ID 列表
        输出: y [num_tokens, embedding_dim] — embedding 向量

        逻辑:
          1. 判断每个 token_id 是否在本 rank 的词表范围内
          2. 不在范围内的 → mask 为 0，映射后的 id 不重要
          3. 在范围内的 → mask 为 1，token_id 减去偏移
          4. 各 rank 查表，mask 乘 embedding
          5. all_reduce 求和：只有一个 rank 有非零值，其余为 0
        """
        if self.tp_size > 1:
            # 判断哪些 token 属于本 rank
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将全局 token_id 转换为本地索引
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight)     # 查表

        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y       # mask 为 0 的 token → embedding 为 0
            dist.all_reduce(y)              # 求和：只有一个 rank 有非零值

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    输出投影层（LM Head）。

    与 Embedding 共享权重（tie_word_embeddings），但 forward 行为不同:
      - Embedding: token_id → embedding（查表 + all_reduce 求和）
      - LM Head:   hidden_states → logits（矩阵乘法 + gather 拼接）

    为什么 LM Head 用 gather 而不是 all_reduce？
      因为采样需要完整的词表概率分布。
      gather 将所有 rank 的部分 logits 拼成完整词表。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [num_tokens, hidden_size]

        prefill 优化: 只取每个序列最后一个位置的 hidden_states
          prefill 时输入是 [t0, t1, t2 | t3, t4]（两个序列）
          但只需要预测 t2 和 t4 的下一个 token
          所以只取 cu_seqlens_q[1:]-1 位置的 hidden_states
        """
        context = get_context()
        if context.is_prefill:
            # 只保留每个序列最后一个位置: [seq0_last, seq1_last, ...]
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # 矩阵乘法: [num_seqs, hidden] @ [vocab/tp, hidden]^T → [num_seqs, vocab/tp]
        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            # gather: 各 rank 的部分 logits 拼成完整词表
            # rank 0 收齐所有 rank 的结果，cat 拼接
            # rank 1+ 的结果设为 None
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] \
                         if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits
