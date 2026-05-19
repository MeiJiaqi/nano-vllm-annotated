"""
SamplingParams — 采样参数
=========================
控制文本生成时的随机性和生成长度。
"""

from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0   # 温度：越高越随机，越低越确定（>0，不支持 greedy=0）
    max_tokens: int = 64       # 最多生成多少个 token（不包含 prompt）
    ignore_eos: bool = False   # True=忽略 EOS 一直生成到 max_tokens（benchmark 用）

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        # Nano-vLLM 当前只支持随机采样，temperature 必须 > 0
