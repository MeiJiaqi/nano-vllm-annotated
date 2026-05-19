<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM (代码注释版)

> 这是 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 fork，仅用于个人学习。
>
> 每行代码都添加了详细的中文注释，帮助理解 LLM 推理框架的核心原理。
>
> 原项目是一个非常优秀的 vLLM 精简实现，仅 ~1200 行 Python 就涵盖了 PagedAttention、Prefix Caching、Continuous Batching、CUDA Graph、Tensor Parallelism 等关键优化技术。强烈推荐给想学习推理框架的同学。

## 与原仓库的区别

- **每行代码附带中文注释**，解释设计意图和实现原理
- **文件结构完全一致**，方便对照原仓库阅读
- **未修改任何功能代码**，只是添加了注释
- **额外附带一份 [TUTORIAL.md](TUTORIAL.md) 教程**，从架构全局视角讲解各组件的作用

## 阅读顺序建议

| 优先级 | 文件 | 内容 |
|--------|------|------|
| 1 | [nanovllm/engine/llm_engine.py](nanovllm/engine/llm_engine.py) | 推理引擎主循环 |
| 2 | [nanovllm/engine/sequence.py](nanovllm/engine/sequence.py) | 序列抽象 |
| 3 | [nanovllm/engine/block_manager.py](nanovllm/engine/block_manager.py) | KV Cache + Prefix Caching |
| 4 | [nanovllm/engine/scheduler.py](nanovllm/engine/scheduler.py) | 调度器 |
| 5 | [nanovllm/engine/model_runner.py](nanovllm/engine/model_runner.py) | 模型执行 / CUDA Graph / TP |
| 6 | [nanovllm/models/qwen3.py](nanovllm/models/qwen3.py) | Qwen3 模型结构 |
| 7 | [nanovllm/layers/](nanovllm/layers/) | 各算子层实现 |

---

> 以下为原仓库 README 内容。

---

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
