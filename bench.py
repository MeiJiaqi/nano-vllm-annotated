"""
Benchmark — 性能测试（带注释）
==============================
对比 Nano-vLLM 和 vLLM 的吞吐量。

测试配置:
  - 256 个请求
  - 输入长度随机 100~1024 token
  - 输出长度随机 100~1024 token
  - ignore_eos=True（忽略 EOS，固定生成指定长度，保证公平对比）

用法:
  将注释的 vllm import 和 dict 格式取消注释即可对比 vLLM。
"""

import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams  # 取消注释以对比 vLLM


def main():
    seed(0)
    num_seqs = 256          # 总请求数
    max_input_len = 1024    # 最大输入长度
    max_ouput_len = 1024    # 最大输出长度

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # enforce_eager=False: 启用 CUDA Graph（benchmark 追求最大吞吐）
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 生成随机测试数据
    # 每个 prompt 是随机 token_id 序列（不需要有意义，只测速度）
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,  # 忽略 EOS，固定生成长度
            max_tokens=randint(100, max_ouput_len),
        )
        for _ in range(num_seqs)
    ]

    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 预热: 先跑一次，触发 CUDA kernel 编译
    llm.generate(["Benchmark: "], SamplingParams())

    # 正式 benchmark
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)

    # 计算吞吐量
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
