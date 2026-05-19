"""
Config — 全局配置
=================
所有推理相关的超参数都在这里，通过 dataclass 管理。
LLMEngine 初始化时会把用户传入的 kwargs 过滤后传给 Config。
"""

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)  # slots=True 节省内存，禁止动态添加属性
class Config:
    # —— 模型路径（必填）——
    model: str                           # 本地 HuggingFace 模型目录路径

    # —— 批处理限制 ——
    max_num_batched_tokens: int = 16384  # 每轮 step 最多处理多少 token（prefill 的 token budget）
    max_num_seqs: int = 512              # 同时处理的最大序列数（batch size 上限）

    # —— 模型长度 & 显存 ——
    max_model_len: int = 4096            # 单个序列最大长度（受模型上下文窗口限制）
    gpu_memory_utilization: float = 0.9  # GPU 显存使用比例（0.9 = 90%，留 10% 余量）

    # —— 并行 & 执行模式 ——
    tensor_parallel_size: int = 1        # 张量并行 GPU 数（1=单卡，2/4/8=多卡）
    enforce_eager: bool = False          # True=禁用 CUDA Graph（方便调试），False=启用

    # —— 以下由代码自动填充，用户一般不需要传 ——
    hf_config: AutoConfig | None = None  # HuggingFace 模型配置（自动从 model 路径加载）
    eos: int = -1                        # EOS token ID（自动从 tokenizer 获取）
    kvcache_block_size: int = 256        # KV Cache 每个 block 容纳的 token 数（必须是 256 的倍数）
    num_kvcache_blocks: int = -1         # KV Cache block 总数（运行时根据显存计算）

    def __post_init__(self):
        """dataclass 初始化后自动调用，做校验和自动推导"""
        assert os.path.isdir(self.model)                          # 模型路径必须存在
        assert self.kvcache_block_size % 256 == 0                 # block_size 对齐要求
        assert 1 <= self.tensor_parallel_size <= 8                # TP 规模限制

        # 从 HuggingFace config.json 自动加载模型架构参数
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # 如果用户指定的 max_model_len 超过模型本身限制，自动修正
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
