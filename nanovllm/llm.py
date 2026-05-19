"""
LLM — 用户 API 入口
===================
vLLM 兼容的接口层，直接继承 LLMEngine。
用户只需: LLM(path) → llm.generate(prompts, sampling_params)
"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    对 LLMEngine 的薄封装，保持 vLLM 的用户接口习惯。

    使用示例:
        llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)
        outputs = llm.generate(["Hello!"], SamplingParams(temperature=0.6))
    """
    pass
