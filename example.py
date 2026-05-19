"""
Example — 使用示例（带注释）
============================
演示 Nano-vLLM 的基本用法:
  1. 加载模型和分词器
  2. 设置采样参数
  3. 准备输入（支持 chat template）
  4. 调用 generate
  5. 打印结果
"""

import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # —— 1. 加载模型 ——
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # enforce_eager=True: 禁用 CUDA Graph（调试时推荐）
    # tensor_parallel_size=1: 单卡推理
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # —— 2. 设置采样参数 ——
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # —— 3. 准备输入 ——
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 应用 chat template（Qwen3 的对话格式）
    # tokenize=False: 此时不 tokenize，等 LLMEngine 内部再 tokenize
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,  # 添加生成提示（如 "<|assistant|>"）
        )
        for prompt in prompts
    ]

    # —— 4. 生成 ——
    outputs = llm.generate(prompts, sampling_params)

    # —— 5. 打印结果 ——
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
