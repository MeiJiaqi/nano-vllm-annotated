"""
Qwen3 模型实现
==============
Nano-vLLM 目前支持 Qwen3 架构。
模块化设计使其可以方便扩展支持其他模型（如 LLaMA、Mistral）。

模型结构:
  Qwen3ForCausalLM
  ├── Qwen3Model
  │   ├── VocabParallelEmbedding          ← 词嵌入（支持 TP）
  │   ├── Qwen3DecoderLayer × N
  │   │   ├── input_layernorm (RMSNorm)   ← Pre-Attention Norm
  │   │   ├── Qwen3Attention
  │   │   │   ├── QKVParallelLinear       ← Q、K、V 合并投影（支持 TP）
  │   │   │   ├── Q/K Norm (RMSNorm)      ← Qwen3 特有：QK 归一化
  │   │   │   ├── RotaryEmbedding         ← RoPE 位置编码
  │   │   │   ├── Attention               ← Flash Attention + KV Cache
  │   │   │   └── RowParallelLinear       ← O 投影（支持 TP）
  │   │   ├── post_attention_layernorm    ← Pre-MLP Norm
  │   │   └── Qwen3MLP
  │   │       ├── MergedColumnParallelLinear  ← gate+up 合并投影
  │   │       ├── SiluAndMul                 ← SiLU 门控激活
  │   │       └── RowParallelLinear         ← down 投影
  │   └── RMSNorm (final norm)
  └── ParallelLMHead                     ← 输出投影（与 embedding 共享权重）

残差连接的融合优化:
  每个 DecoderLayer 的 forward 接受 (hidden_states, residual):
    - RMSNorm 在归一化前先将 x + residual，新 residual = 相加结果
    - 这样将 "残差加法" 和 "RMSNorm" 融合为一个 kernel，减少显存读写
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# ============================================================
#  Qwen3Attention — 单层 Attention
# ============================================================
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,         # GQA: KV heads 可以少于 Q heads
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,    # Qwen3 使用 QK Norm 替代 bias
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()

        # —— 计算 TP 切分后的 head 数 ——
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size          # 当前 rank 的 Q head 数

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size    # 当前 rank 的 KV head 数

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim              # Q 的维度
        self.kv_size = self.num_kv_heads * self.head_dim          # K/V 的维度
        self.scaling = self.head_dim ** -0.5                      # Attention 缩放因子

        # —— QKV 合并投影（TP 切分）——
        # 将 Q、K、V 三个投影合并为一个矩阵乘法: [Wq | Wk | Wv]
        # 好处: 一次矩阵乘法代替三次，减少 kernel launch
        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim,
            self.total_num_heads, self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # —— O 投影（TP 切分，需要 all-reduce）——
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size, bias=False,
        )

        # —— RoPE 位置编码 ——
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim,
            max_position=max_position, base=rope_theta,
        )

        # —— Flash Attention + KV Cache ——
        self.attn = Attention(self.num_heads, self.head_dim,
                              self.scaling, self.num_kv_heads)

        # —— QK 归一化（Qwen3 特有）——
        # Qwen3 对 Q 和 K 分别做 RMSNorm，替代传统的 qkv_bias
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,       # [num_tokens] 各 token 的位置索引
        hidden_states: torch.Tensor,   # [num_tokens, hidden_size] 输入
    ) -> torch.Tensor:
        # 1. QKV 投影: [num_tokens, hidden] → [num_tokens, q_size + 2*kv_size]
        qkv = self.qkv_proj(hidden_states)

        # 2. 拆分 Q、K、V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 3. Reshape 为多头格式: [num_tokens, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 4. QK 归一化（Qwen3 特有）
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 5. RoPE 位置编码
        q, k = self.rotary_emb(positions, q, k)

        # 6. Attention（内含 KV Cache 读写）
        o = self.attn(q, k, v)

        # 7. O 投影: [num_tokens, num_heads*head_dim] → [num_tokens, hidden]
        output = self.o_proj(o.flatten(1, -1))
        return output


# ============================================================
#  Qwen3MLP — 前馈网络（SwiGLU 架构）
# ============================================================
class Qwen3MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()

        # gate_proj 和 up_proj 合并为一个矩阵乘法
        # gate: 门控信号, up: 升维投影
        # 输出 = down_proj( SiLU(gate(x)) * up(x) )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()  # SiLU 门控 + 逐元素乘法

    def forward(self, x):
        gate_up = self.gate_up_proj(x)  # [N, 2*intermediate]
        x = self.act_fn(gate_up)        # [N, intermediate]  (SiLU(gate) * up)
        x = self.down_proj(x)           # [N, hidden]
        return x


# ============================================================
#  Qwen3DecoderLayer — 单个 Transformer 层
# ============================================================
class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        一次 Transformer 层的前向。

        参数:
            positions:     位置编码索引
            hidden_states: 当前层的输入
            residual:      上一层的残差（None 表示这是第一层）

        返回:
            (hidden_states, residual): 下一层的输入和新的残差

        融合残差的设计:
          传统写法:
            x = x + attention(norm(x))
            x = x + mlp(norm(x))
          这里的写法:
            x, residual = norm(x, residual)     # norm 内部完成 residual add
            x = attention(x)
            x, residual = norm(x, residual)     # 同上
            x = mlp(x)
          好处: norm kernel 内部同时做 add + norm，减少显存读写
        """
        # —— Pre-Attention Norm + 残差融合 ——
        if residual is None:
            # 第一层: 没有历史残差，只做 norm
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层: x = norm(x + residual)，新 residual = x + residual
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # —— Attention ——
        hidden_states = self.self_attn(positions, hidden_states)

        # —— Post-Attention Norm + 残差融合 ——
        # 将 attention 输出加回 residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # —— MLP ——
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ============================================================
#  Qwen3Model — 完整的 Transformer 主干
# ============================================================
class Qwen3Model(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,   # [num_tokens] token ID
        positions: torch.Tensor,   # [num_tokens] 位置索引
    ) -> torch.Tensor:
        # 1. Embedding
        hidden_states = self.embed_tokens(input_ids)

        # 2. 逐层 Transformer
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 3. 最终 LayerNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ============================================================
#  Qwen3ForCausalLM — 完整的因果语言模型
# ============================================================
class Qwen3ForCausalLM(nn.Module):
    """
    完整的 Qwen3 模型，包含 Embedding + Transformer + LM Head。

    packed_modules_mapping:
      将 HuggingFace 的标准权重名映射到 Nano-vLLM 的合并层。
      例如:
        "model.layers.0.self_attn.q_proj.weight" → qkv_proj 的 "q" 部分
        "model.layers.0.mlp.gate_proj.weight"   → gate_up_proj 的第 0 部分
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),      # Q 权重合并到 QKV
        "k_proj": ("qkv_proj", "k"),      # K 权重合并到 QKV
        "v_proj": ("qkv_proj", "v"),      # V 权重合并到 QKV
        "gate_proj": ("gate_up_proj", 0), # gate 合并到 gate+up（第 0 个分片）
        "up_proj": ("gate_up_proj", 1),   # up 合并到 gate+up（第 1 个分片）
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 权重绑定: lm_head 和 embed_tokens 共享权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """返回所有位置的 hidden_states（prefill 用于进一步处理）"""
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        从 hidden_states 计算 logits。

        prefill 时只取每个序列最后一个位置的 hidden_states 来计算 logits
        （因为只需要预测下一个 token，不需要中间位置的 logits）
        decode 时只有一个 token，直接计算。
        """
        return self.lm_head(hidden_states)
