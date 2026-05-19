"""
Attention — Flash Attention + KV Cache 管理
=============================================
这是实际执行 attention 计算的层，包含:
1. 将计算出的 K、V 写入 KV Cache（Triton kernel）
2. Prefill: 使用 flash_attn_varlen_func（变长序列 flash attention）
3. Decode:  使用 flash_attn_with_kvcache（从 KV Cache 读取）

KV Cache 的物理布局:
  k_cache / v_cache 的 shape: [num_blocks, block_size, num_kv_heads, head_dim]
  slot_mapping [num_tokens]: 每个 token 的 K/V 应该写入哪个 slot:
    slot = block_id * block_size + offset_in_block
  -1 表示该 token 不需要写 KV Cache（CUDA Graph 的 padding token）
"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# ============================================================
#  Triton Kernel: 将 K、V 写入 KV Cache
# ============================================================
@triton.jit
def store_kvcache_kernel(
    key_ptr,           # 当前计算出的 K: [N, num_heads, head_dim]
    key_stride,        # K 的第一个维度的 stride
    value_ptr,         # 当前计算出的 V: [N, num_heads, head_dim]
    value_stride,      # V 的第一个维度的 stride
    k_cache_ptr,       # GPU 上的 K Cache: [num_blocks*block_size, num_heads, head_dim]
    v_cache_ptr,       # GPU 上的 V Cache
    slot_mapping_ptr,  # [N]: 每个 token 写入 KV Cache 的 slot 位置
    D: tl.constexpr,   # num_heads * head_dim（每个 token 需要复制的元素数）
):
    """
    每个 Triton program 处理一个 token（N 个 program 并行）。

    对于 token idx:
      1. 从 slot_mapping 读取它的目标 slot
      2. 如果 slot == -1（无效 token），直接返回
      3. 将 K[idx] 复制到 k_cache[slot]
      4. 将 V[idx] 复制到 v_cache[slot]
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return  # 无效 token（CUDA Graph padding），直接跳过

    # 计算偏移量
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 读取 K[idx] 和 V[idx]
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 写入 KV Cache
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,          # [N, num_heads, head_dim] 当前计算的 K
    value: torch.Tensor,        # [N, num_heads, head_dim] 当前计算的 V
    k_cache: torch.Tensor,      # [num_blocks*block_size, num_heads, head_dim] GPU K Cache
    v_cache: torch.Tensor,      # GPU V Cache
    slot_mapping: torch.Tensor, # [N] 每个 token 的目标 slot 位置
):
    """
    将当前 batch 计算的 K、V 写入持久化的 KV Cache。

    使用 Triton kernel 实现高效的并行写入。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 每个 token 需要写入的元素总数

    # stride 校验：确保 tensor 是连续布局
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 启动 N 个 Triton program（每个 token 一个）
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D,
    )


# ============================================================
#  Attention 层
# ============================================================
class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale                # 1/sqrt(head_dim)，缩放因子
        self.num_kv_heads = num_kv_heads  # GQA: KV heads 可以少于 Q heads

        # KV Cache 引用（在 allocate_kv_cache() 中被赋值为实际的内存切片）
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, head_dim]        Query
        k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]     Key
        v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]     Value
    ):
        # 从全局 Context 读取当前 step 的元信息
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # —— 步骤 1: 将 K、V 写入 KV Cache ——
        if k_cache.numel() and v_cache.numel():
            # 按 slot_mapping 将 K/V 写入对应位置
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # —— 步骤 2: 执行 Attention ——
        if context.is_prefill:
            # === Prefill: 变长序列 Flash Attention ===
            # 使用 cu_seqlens_q/k 支持变长序列，一次 kernel 处理所有 token
            if context.block_tables is not None:
                # Prefix cache 场景: K/V 不在连续内存中，需要 block_table 间接寻址
                # 此时 k,v 用 KV Cache 中的值（因为部分 token 的 KV 可能在别的 block）
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,                  # 因果 mask: token_i 只能看到 token_0..i
                block_table=context.block_tables,  # prefix cache 时需要
            )
        else:
            # === Decode: 带 KV Cache 的 Flash Attention ===
            # 每个 seq 只有 1 个新 token 的 Q，但 K/V 要关注所有历史 token
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),              # [batch, 1, num_heads, head_dim]
                k_cache,                     # [num_blocks*block_size, num_kv_heads, head_dim]
                v_cache,
                cache_seqlens=context.context_lens,  # 每个序列的当前长度
                block_table=context.block_tables,    # 逻辑→物理 block 映射
                softmax_scale=self.scale,
                causal=True,
            )

        return o
