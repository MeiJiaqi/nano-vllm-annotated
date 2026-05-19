"""
Context — 全局上下文
====================
在模块层级维护一个全局 Context 对象，各层通过 get_context() 获取。

为什么用全局 Context 而不是传参？
  Attention 层需要知道 prefill/decode 阶段、slot_mapping、block_tables 等信息。
  但这些信息不是标准的 nn.Module.forward 参数。
  全局 Context 是一种在保持模型代码整洁的同时传递元信息的折中方案。

使用流程:
  1. ModelRunner 在模型 forward 前调用 set_context()
  2. 各层（Attention、LM Head）通过 get_context() 读取
  3. 模型 forward 完成后 ModelRunner 调用 reset_context() 清空
"""

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """一次 step 的全局元信息"""

    # 阶段标记
    is_prefill: bool = False         # True=prefill, False=decode

    # Prefill 专用
    cu_seqlens_q: torch.Tensor | None = None   # [num_seqs+1] Q 累积长度
    cu_seqlens_k: torch.Tensor | None = None   # [num_seqs+1] K 累积长度
    max_seqlen_q: int = 0                      # 最长 Q 序列
    max_seqlen_k: int = 0                      # 最长 K 序列（prefix cache 时可能 > max_seqlen_q）

    # 通用
    slot_mapping: torch.Tensor | None = None   # [num_tokens] → KV Cache slot
    context_lens: torch.Tensor | None = None   # [num_seqs] 各序列当前长度（decode）
    block_tables: torch.Tensor | None = None   # [num_seqs, max_blocks] block 映射表


# 模块级单例
_CONTEXT = Context()


def get_context():
    """获取当前 step 的 Context"""
    return _CONTEXT


def set_context(is_prefill,
                cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None):
    """设置当前 step 的 Context（ModelRunner 调用）"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables)


def reset_context():
    """清空 Context（step 结束后调用）"""
    global _CONTEXT
    _CONTEXT = Context()
