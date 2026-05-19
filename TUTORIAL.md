# Nano-vLLM 推理框架教程

本教程基于对 Nano-vLLM 源码（~1200 行）的深入分析，从零讲解 LLM 推理引擎的核心架构与关键优化技术。

---

## 目录

1. [为什么要做推理框架？](#1-为什么要做推理框架)
2. [整体架构一览](#2-整体架构一览)
3. [请求的生命周期](#3-请求的生命周期)
4. [核心组件详解](#4-核心组件详解)
   - [4.1 Sequence：序列抽象](#41-sequence序列抽象)
   - [4.2 BlockManager：KV Cache 管理与 Prefix Caching](#42-blockmanagerkv-cache-管理与-prefix-caching)
   - [4.3 Scheduler：调度器](#43-scheduler调度器)
   - [4.4 ModelRunner：模型执行器](#44-modelrunner模型执行器)
5. [关键优化技术](#5-关键优化技术)
   - [5.1 Continuous Batching（连续批处理）](#51-continuous-batching连续批处理)
   - [5.2 PagedAttention / Block-based KV Cache](#52-pagedattention--block-based-kv-cache)
   - [5.3 Prefix Caching（前缀缓存）](#53-prefix-caching前缀缓存)
   - [5.4 Chunked Prefill（分块预填充）](#54-chunked-prefill分块预填充)
   - [5.5 CUDA Graph](#55-cuda-graph)
   - [5.6 Tensor Parallelism（张量并行）](#56-tensor-parallelism张量并行)
6. [模型实现](#6-模型实现)
7. [完整数据流示例](#7-完整数据流示例)
8. [如何扩展 Nano-vLLM](#8-如何扩展-nano-vllm)
9. [总结与学习路线](#9-总结与学习路线)

---

## 1. 为什么要做推理框架？

直接用 HuggingFace Transformers 做推理不行吗？不行的原因有三：

**问题一：KV Cache 内存浪费**
每个 token 生成时，Attention 层都需要计算所有历史 token 的 Key/Value。如果每次都重新计算，复杂度是 O(n²)。正确做法是缓存已计算的 KV，但 HuggingFace 的实现按 max_length 预分配连续显存，大部分空间被浪费。

**问题二：无法批量处理不同长度的请求**
实际服务中同时有 100 个请求，有的刚进来（prefill 阶段，需要并行处理大量 prompt token），有的在逐 token 生成（decode 阶段，每次只需要算 1 个 token）。HuggingFace 的 batch 要求所有序列等长（或做 padding），无法高效混合处理。

**问题三：静态显存分配**
提前为 "最大 batch size × 最大序列长度" 分配 KV Cache 显存，实际使用中大部分闲置。

vLLM 解决这三个问题的核心思路是：**PagedAttention**（像操作系统管理虚拟内存一样管理 KV Cache）+ **Continuous Batching**（动态调度不同阶段的请求）。

---

## 2. 整体架构一览

```
┌────────────────────────────────────────────────────┐
│                    LLM (用户 API)                     │
│                 继承自 LLMEngine                      │
└────────────────────┬───────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────┐
│                  LLMEngine                          │
│  generate() → 循环 { step() } → 返回结果             │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│  │ Scheduler │  │BlockMgr  │  │ ModelRunner  │     │
│  │ (调度器)   │  │(显存管理) │  │ (模型执行)    │     │
│  └──────────┘  └──────────┘  └──────────────┘     │
└────────────────────────────────────────────────────┘

文件结构：
  nanovllm/
  ├── llm.py              # 用户 API 入口
  ├── config.py           # 配置管理
  ├── sampling_params.py  # 采样参数
  ├── engine/
  │   ├── llm_engine.py   # 引擎主循环
  │   ├── sequence.py     # 序列抽象
  │   ├── scheduler.py    # 调度器
  │   ├── block_manager.py# KV Cache 块管理 + Prefix Caching
  │   └── model_runner.py # 模型执行 + CUDA Graph + TP
  ├── models/
  │   └── qwen3.py        # Qwen3 模型定义
  ├── layers/
  │   ├── attention.py    # Flash Attention + KV Cache
  │   ├── linear.py       # 并行线性层 (TP)
  │   ├── layernorm.py    # RMSNorm
  │   ├── rotary_embedding.py # RoPE
  │   ├── activation.py   # SiLU 激活
  │   ├── embed_head.py   # Embedding & LM Head
  │   └── sampler.py      # 采样器
  └── utils/
      ├── context.py      # 全局上下文 (传递 prefill/decode 信息)
      └── loader.py       # 模型权重加载
```

三条核心设计原则：
1. **分离调度与执行**：Scheduler 决定 "跑哪些序列、跑多少 token"，ModelRunner 只负责执行
2. **Block 化管理 KV Cache**：KV Cache 按固定大小（256 token）切分为 Block，通过 Block Table 映射
3. **全局 Context**：各层通过 `get_context()` 获取当前 step 的元信息（prefill/decode、slot_mapping、block_tables 等）

---

## 3. 请求的生命周期

一个请求从进入到返回的完整流程如下：

```
1. 用户调用 llm.generate(prompts, sampling_params)
       │
2.     ├─ 每个 prompt 被 encode 为 token_ids
       ├─ 创建 Sequence 对象，状态 = WAITING
       └─ 加入 Scheduler.waiting 队列
              │
3.     ┌─ step() 循环 ────────────────────────┐
       │                                       │
       │  Scheduler.schedule()                 │
       │    ├─ 从 waiting 取出 seqs（prefill） │
       │    ├─ 检查 BlockManager 能否分配 KV   │
       │    ├─ 分配 block_table                │
       │    └─ 从 running 取出 seqs（decode）  │
       │                                       │
       │  ModelRunner.run(seqs, is_prefill)    │
       │    ├─ prepare_prefill / prepare_decode│
       │    ├─ 设置 Context                    │
       │    ├─ Model.forward → logits          │
       │    └─ Sampler → token_ids             │
       │                                       │
       │  Scheduler.postprocess()              │
       │    ├─ 哈希 block 内容 (prefix cache)  │
       │    ├─ 更新 seq 状态                   │
       │    └─ FINISHED → deallocate           │
       │                                       │
       └──────────────────────────────────────┘
              │
4. 所有 seqs FINISHED → tokenizer.decode → 返回
```

---

## 4. 核心组件详解

### 4.1 Sequence：序列抽象

**[sequence.py](nanovllm/engine/sequence.py)**

Sequence 代表一个推理请求的完整状态：

```python
class Sequence:
    block_size = 256            # 类变量，每个 block 容纳的 token 数
    counter = count()           # 自增 ID 分配器

    seq_id: int                 # 唯一标识
    status: SequenceStatus      # WAITING / RUNNING / FINISHED
    token_ids: list[int]        # 完整 token 序列 (prompt + 已生成的)
    num_tokens: int             # 当前总 token 数
    num_prompt_tokens: int      # prompt 部分的 token 数
    num_cached_tokens: int      # 已有缓存的 token 数（命中 prefix cache）
    num_scheduled_tokens: int   # 本轮调度的 token 数
    is_prefill: bool            # 是否处于 prefill 阶段
    block_table: list[int]      # KV Cache block ID 列表
    temperature: float          # 采样温度
    max_tokens: int             # 最大生成 token 数
```

**关键概念**：
- `num_cached_tokens`：该序列中已被 prefix cache 覆盖的 token 数量。这些 token 的 KV Cache 不需要重新计算
- `num_scheduled_tokens`：本次 step 要处理多少 token（prefill 时 >1，decode 时 =1）
- `block_table`：类似操作系统的页表，将逻辑 token 位置映射到物理 KV Cache block

### 4.2 BlockManager：KV Cache 管理与 Prefix Caching

**[block_manager.py](nanovllm/engine/block_manager.py)**

这是整个系统最精妙的部分，核心思想来自 vLLM 的 PagedAttention。

**Block 结构**：
```python
class Block:
    block_id: int       # 物理块 ID
    ref_count: int      # 引用计数（多个 seq 可共享同一 block）
    hash: int           # 内容的 xxhash64 哈希值
    token_ids: list     # 该 block 对应的 token
```

**BlockManager 的核心数据结构**：
```python
class BlockManager:
    blocks: list[Block]              # 所有物理 block
    hash_to_block_id: dict[int,int]  # 哈希 → block 的映射（prefix cache 的核心）
    free_block_ids: deque            # 空闲 block 队列
    used_block_ids: set              # 已使用 block 集合
```

#### Prefix Caching 算法

这是 Nano-vLLM 最值得深入理解的优化，实现在 `can_allocate()` 中：

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)               # 取第 i 个 block 的 token
        h = self.compute_hash(token_ids, h)    # 链式哈希：h = hash(token_ids, prev_h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                               # 不匹配则停止
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1                 # 已在显存中，不需重新分配
    if len(self.free_block_ids) < num_new_blocks:
        return -1                               # 显存不足
    return num_cached_blocks
```

**链式哈希** 是精髓：每个 block 的哈希 = hash(当前 block 的 token, 前一个 block 的哈希)。这保证了相同前缀产生相同的哈希链，而不用存储完整 token 序列做比较。

**具体例子**：
- 请求 A：`"Hello, how are you"` → block0 hash=H1, block1 hash=H2
- 请求 B：`"Hello, how are you today?"` → block0 hash=H1 ✓（命中），block1 hash=H2 ✓（命中），block2 token 不同，哈希不匹配
- B 可以直接复用 A 的前 2 个 block 的 KV Cache，只需为 block2 分配新显存

**allocate / deallocate 与引用计数**：
- `allocate`：命中缓存的 block ref_count++，新 block 从 free 队列取
- `deallocate`：所有 block ref_count--，减到 0 的归还 free 队列（但哈希映射保留，后续请求仍可命中）
- `hash_blocks`：在 postprocess 中调用，计算并记录已完成 block 的哈希到 `hash_to_block_id`

### 4.3 Scheduler：调度器

**[scheduler.py](nanovllm/engine/scheduler.py)**

调度器维护两个队列：`waiting`（等待 prefill）和 `running`（逐 token decode）。核心在 `schedule()` 方法：

**调度策略**（两阶段）：

```
阶段1 — Prefill（优先）：
  while waiting 非空 且 未达 max_num_seqs:
      seq = waiting[0]
      检查剩余 token_budget 是否够
      检查 BlockManager 能否分配 KV Cache
      分配 block_table
      确定 num_scheduled_tokens（支持 chunked prefill）
      标记为 RUNNING，移入 running 队列

  返回 scheduled_seqs, is_prefill=True

阶段2 — Decode（无 prefill 可调度时）：
  while running 非空 且 未达 max_num_seqs:
      seq = running.popleft()
      检查 Can_append（是否需要新 block）
      若不满足：preempt（抢占换出）——将 seq 移回 waiting
      num_scheduled_tokens = 1
      返回 scheduled_seqs, is_prefill=False
```

**关键设计决策**：

1. **Prefill 优先**：每次 step 先尝试做 prefill，减少首 token 延迟
2. **Chunked Prefill**：见下方 [5.4 节](#54-chunked-prefill分块预填充)
3. **抢占机制**：decode 阶段显存不足时，将序列的 KV Cache 释放并移回 waiting 队列，下次重新 prefill（因为 KV Cache 被释放了）

### 4.4 ModelRunner：模型执行器

**[model_runner.py](nanovllm/engine/model_runner.py)**

ModelRunner 负责将 Scheduler 的调度决策转化为实际的 GPU 计算。

**初始化流程**：
```
__init__:
  1. 初始化 NCCL 进程组 (Tensor Parallelism)
  2. 构建 Qwen3ForCausalLM 模型
  3. 加载权重 (safetensors)
  4. warmup_model() — 跑一次 prefill 预热 CUDA
  5. allocate_kv_cache() — 计算可用显存，分配 KV Cache tensor
  6. capture_cudagraph() — 捕获 CUDA Graph（如果 enforce_eager=False）
  7. 非 rank0 进入 loop() 等待主进程指令
```

#### Prefill 阶段的数据准备

```python
def prepare_prefill(self, seqs):
    # 为每个序列收集：
    #   input_ids  — 从 seq[start:end] 取 token
    #   positions  — 位置编码
    #   cu_seqlens_q/k — Flash Attention 需要的累积序列长度
    #   slot_mapping — token 在 KV Cache 中的物理位置
    #   block_tables — 当 prefix cache 命中时，需要 block_table 做间接寻址
```

**slot_mapping 的计算** 是一个关键细节：每个 token 需要知道自己在该序列 block_table 中的确切位置，这样 attention kernel 才能正确读写 KV Cache。

- 例：如果 block_size=256，block_table=[3, 7]，处理 token 位置 300
  - 300 // 256 = 1 → 在 block_table[1] = 7 号物理 block
  - slot = 7 * 256 + (300 % 256) = 1792 + 44 = 1836

#### Decode 阶段的数据准备

```python
def prepare_decode(self, seqs):
    # decode 阶段每次每个 seq 只处理 1 个 token
    # input_ids: seq.last_token（最后一个 token）
    # positions: len(seq) - 1
    # slot_mapping: block_table[-1] * block_size + last_block_num_tokens - 1
    # context_lens: 用于 flash_attn_with_kvcache 的 cache_seqlens 参数
```

#### Context 全局状态

[context.py](nanovllm/utils/context.py) 实现了一个模块级的全局 Context，各层通过 `get_context()` 获取当前 step 的类型和元数据：

```python
@dataclass
class Context:
    is_prefill: bool
    cu_seqlens_q / cu_seqlens_k: Tensor  # flash attention varlen 参数
    max_seqlen_q / max_seqlen_k: int     # 最大序列长度
    slot_mapping: Tensor                 # token → KV Cache slot 映射
    context_lens: Tensor                 # 各序列当前长度 (decode)
    block_tables: Tensor                 # 各序列的 block table
```

**为什么用全局 Context 而不是传参？** 因为 attention、embed_head 等层需要 prefill/decode 相关的元信息，而标准的 `nn.Module.forward` 接口不方便传递这些额外参数。全局 Context 是一种在保持模型代码整洁的同时传递元信息的折中方案。

---

## 5. 关键优化技术

### 5.1 Continuous Batching（连续批处理）

**问题**：传统做法是一次性把一个 batch 的所有请求处理完再接收新请求。但 LLM 推理中，不同请求的生成长度差异极大——有的 10 个 token 就结束了，有的要 1000 个。

**方案**：每个 step 动态决定 batch 中跑哪些序列。Nano-vLLM 在 Scheduler 中每个 step 都从 waiting/running 队列中重新挑选。序列完成后可以立即从 running 队列移出，腾出空间给新序列。

**收益**：相比 static batching，吞吐量提升数倍。

### 5.2 PagedAttention / Block-based KV Cache

**问题**：KV Cache 的显存如何管理？如果为每个序列分配连续的最大长度显存，会造成严重浪费（内部碎片 + 预留未使用的空间）。

**方案**：参考操作系统的虚拟内存分页：

```
逻辑视角 (每个 Sequence)         物理视角 (GPU 显存)
   token 0-255   → block_id 3  ────→  Block 0
   token 256-511 → block_id 7  ────→  Block 1
   token 512-767 → block_id 2  ────→  Block 2
                                    →  Block 3 ← 被另一个 seq 引用
                                    →  ...
```

**实现要点**：
- `block_size = 256`（必须是 256 的倍数，保证对齐）
- `block_table: list[int]` 维护逻辑→物理映射
- 按需分配：只分配实际需要的 block 数量
- 引用计数：多个序列可共享同一 block（prefix cache 的基础）

**显存计算**（`allocate_kv_cache` 中）：
```python
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
num_blocks = (total * gpu_memory_utilization - used) // block_bytes
```

### 5.3 Prefix Caching（前缀缓存）

**问题**：多个请求共享相同前缀（如 system prompt、few-shot examples）。每个请求都重新计算这些公共 token 的 KV Cache 是浪费。

**方案**：基于哈希的 KV Cache 复用（详见 [4.2 节](#42-blockmanagerkv-cache-管理与-prefix-caching)）。

**实现亮点**：
1. **链式哈希**：`hash(block_i) = xxhash64(block_i_tokens, hash(block_{i-1}))`。这保证了前缀匹配的传递性和唯一性
2. **延迟哈希**：block 在 `hash_blocks()` 中才被注册到 `hash_to_block_id`，避免半完成的 block 被错误复用
3. **引用计数共享**：多个 seq 的 block_table 可以指向同一物理 block，ref_count 管理生命周期

### 5.4 Chunked Prefill（分块预填充）

**问题**：一个长 prompt（如 4096 token）的 prefill 需要大量计算。如果一次性 prefill 完，会阻塞其他请求的首 token 生成（TTFT），影响延迟。

**方案**：将长 prefill 拆成多个 chunk，在多个 step 中完成。每个 step 处理一部分 token，穿插 decode。

**实现**（scheduler.py 中的关键逻辑）：
```python
# 只有当这是第一批调度序列时才允许 chunked prefill
if remaining < num_tokens and scheduled_seqs:
    break  # 已经有其他 seq 被调度了，不再分块
seq.num_scheduled_tokens = min(num_tokens, remaining)
# 序列未完全 prefill，保持在 waiting 状态
if seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens:
    continue  # 不将 seq 移到 running
```

这里的约束 "only allow chunked prefill for the first seq" 简化了实现——一次只对一个序列做分块 prefill。

### 5.5 CUDA Graph

**问题**：Decode 阶段每次只处理 1 个 token/batch，kernel launch overhead 占比很大。每个 kernel 启动需要 ~5-10μs，而实际计算可能只需 ~50μs。高频的 kernel launch 成为瓶颈。

**方案**：CUDA Graph 将整个 decode 过程（所有层的 forward pass）录制为一个图，后续只需一次 replay 调用即可执行整个图，消除 kernel launch overhead。

**Nano-vLLM 的实现**（`capture_cudagraph`）：

```python
# 为不同 batch_size 分别录制 CUDA Graph
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    # 用空数据预热
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
    # 录制
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
    self.graphs[bs] = graph
```

**运行时**：根据实际 batch size 选择 ≥ 且最接近的 pre-captured graph，将真实数据填入预留的 tensor 槽位后 replay。

**注意**：
- CUDA Graph 要求输入 shape 固定，因此需要为不同 batch_size 分别录制
- 只用于 decode（input 固定为 1 token/seq），prefill 的输入长度不固定，无法录制
- 设置 `enforce_eager=True` 可禁用，方便调试
- 当 batch_size > 512 时也走 eager 模式（代码中 `input_ids.size(0) > 512`）

### 5.6 Tensor Parallelism（张量并行）

**问题**：单 GPU 显存放不下大模型，需要多 GPU 协同。

**方案**：将权重矩阵按列或按行切分到多个 GPU，通过 NCCL 通信保持数学等价。

**Nano-vLLM 的实现**使用了三种并行线性层：

#### ColumnParallelLinear
将权重按列切分，每个 GPU 持有 `output_dim/tp_size` 列：
```
原始: Y = X @ W (W: [in, out])
TP:   Y_i = X @ W_i (W_i: [in, out/tp_size])
每个 GPU 得到 Y 的一部分，各管各的后续计算
```

#### RowParallelLinear
将权重按行切分，需要在最后做 all-reduce：
```
原始: Y = X @ W (W: [in, out])
TP:   Y_i = X_i @ W_i (W_i: [in/tp_size, out])
Y = all_reduce(Y_i)  ← 恢复完整结果
```

#### QKVParallelLinear
Attention 的特殊处理——将 Q、K、V 的权重合并为一个矩阵：
```
原始: Q = X @ Wq, K = X @ Wk, V = X @ Wv
TP:   合并 [Wq | Wk | Wv] → 一次矩阵乘法
各自按 head 数量切分 Q 的 heads 和 KV heads
```

**TP 进程间通信**：
- Rank 0 作为主进程，负责调度和采样
- Rank 1+ 通过 SharedMemory + Event 等待指令
- `call()` 方法：rank0 序列化方法名和参数到共享内存，通知其他 rank 执行

---

## 6. 模型实现

**[models/qwen3.py](nanovllm/models/qwen3.py)**

Nano-vLLM 目前只支持 Qwen3 架构，但其模块化设计可以方便地扩展支持其他模型。

**模型结构**：
```
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding (embed_tokens)
│   ├── Qwen3DecoderLayer × N (layers)
│   │   ├── RMSNorm (input_layernorm)
│   │   ├── Qwen3Attention (self_attn)
│   │   │   ├── QKVParallelLinear (qkv_proj)
│   │   │   ├── RMSNorm × 2 (q_norm, k_norm) [Qwen3 特有]
│   │   │   ├── RotaryEmbedding (rotary_emb)
│   │   │   ├── Attention (attn)
│   │   │   └── RowParallelLinear (o_proj)
│   │   ├── RMSNorm (post_attention_layernorm)
│   │   └── Qwen3MLP (mlp)
│   │       ├── MergedColumnParallelLinear (gate_up_proj)
│   │       ├── SiluAndMul (act_fn)
│   │       └── RowParallelLinear (down_proj)
│   └── RMSNorm (norm)
└── ParallelLMHead (lm_head)
```

**残差连接的设计**：
Nano-vLLM 采用了一种精致的残差融合策略。在 `Qwen3DecoderLayer.forward` 中：

```python
def forward(self, positions, hidden_states, residual):
    # Step 1: RMSNorm + 保存残差
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
    # Step 2: Attention
    hidden_states = self.self_attn(positions, hidden_states)
    # Step 3: RMSNorm + 残差融合（将 attention 输出加回 residual）
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    # Step 4: MLP
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

这里 `RMSNorm.add_rms_forward` 在归一化之前先将 x 和 residual 相加（`x.float().add_(residual.float())`），然后将相加结果保存为新的 residual。这是一种 **fused residual add + norm** 的优化，减少了内存读写。

**权重加载**（[loader.py](nanovllm/utils/loader.py)）：
使用 `packed_modules_mapping` 处理合并权重的加载：
```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),    # Q 权重合并到 QKV
    "k_proj": ("qkv_proj", "k"),    # K 权重合并到 QKV
    "v_proj": ("qkv_proj", "v"),    # V 权重合并到 QKV
    "gate_proj": ("gate_up_proj", 0),  # gate 合并到 gate+up
    "up_proj": ("gate_up_proj", 1),    # up 合并到 gate+up
}
```

这允许从 HuggingFace 标准格式的 safetensors 文件直接加载权重到合并后的层中。

---

## 7. 完整数据流示例

假设有 2 个请求："Hello world"（token=[101, 202]）和 "Hi there"（token=[303, 404]）：

```
Step 1: Prefill
  Scheduler.schedule():
    → 从 waiting 取 seq_0 (Hello world, 2 tokens)
    → BlockManager.allocate(seq_0): block_table = [0]
    → seq_0.num_scheduled_tokens = 2
    → 从 waiting 取 seq_1 (Hi there, 2 tokens)
    → BlockManager.allocate(seq_1): block_table = [1]
    → seq_1.num_scheduled_tokens = 2
    → 返回 [seq_0, seq_1], is_prefill=True

  ModelRunner.run():
    prepare_prefill() →
      input_ids = [101, 202, 303, 404]
      positions  = [0, 1, 0, 1]
      cu_seqlens_q = [0, 2, 4]
      slot_mapping = [0, 1, 256, 257]  # block0 slot [0,1], block1 slot [0,1]
    Model forward:
      Embedding → 各层 Attention + MLP → logits
    Sampler: 取最后位置的 logits → [next_0, next_1]

  Scheduler.postprocess():
    → seq_0.append_token(next_0)
    → seq_1.append_token(next_1)
    → hash_blocks() 注册 block 哈希

Step 2: Decode
  Scheduler.schedule():
    → waiting 为空
    → 从 running 取 seq_0, seq_1
    → seq_0.num_scheduled_tokens = 1
    → seq_1.num_scheduled_tokens = 1
    → 返回 [seq_0, seq_1], is_prefill=False

  ModelRunner.run():
    prepare_decode() →
      input_ids = [next_0, next_1]
      positions  = [2, 2]
      slot_mapping = [2, 258]
      context_lens = [3, 3]
      block_tables = [[0, -1], [1, -1]]
    Model forward (可能走 CUDA Graph 快速路径):
      各层 Attention: flash_attn_with_kvcache(q, k_cache, v_cache, ...)
      → logits → sampler → [next_next_0, next_next_1]

  ...循环直到所有 seq 完成或达到 max_tokens...
```

---

## 8. 如何扩展 Nano-vLLM

如果你想基于 Nano-vLLM 做自己的推理框架，以下是最有价值的扩展方向：

### 8.1 支持更多模型架构
参考 `qwen3.py` 的模式，添加新模型只需：
1. 实现该模型的 Attention、MLP、DecoderLayer、Model 类
2. 定义 `packed_modules_mapping` 用于权重加载
3. 在 `ModelRunner.__init__` 中根据 `hf_config.architectures` 选择模型

### 8.2 支持在线服务（API Server）
目前只支持离线批量推理。添加一个 FastAPI/ray 层即可变为在线推理服务，核心变化是 `generate()` 需要支持流式返回。

### 8.3 更多采样策略
当前只支持 temperature sampling。可以添加：
- Top-K / Top-P 采样
- Beam search
- 惩罚机制（frequency penalty, presence penalty）

### 8.4 Speculative Decoding（投机解码）
用小模型快速生成 draft tokens，大模型并行验证。可以显著提升 decode 吞吐量（2-3x）。

### 8.5 KV Cache 量化
将 KV Cache 用 INT8/FP8 存储，减少显存占用 2-4x，允许更大的 batch size。

### 8.6 Pipeline Parallelism
当前只有 Tensor Parallelism。对于超大模型（70B+），还需要流水线并行将不同层分配到不同 GPU。

### 8.7 更精细的调度策略
当前是简单的 FCFS 调度。可以引入：
- Priority-based scheduling
- Prefill/decode 分离的 batch
- 基于负载的 adaptive batching

---

## 9. 总结与学习路线

### 学习 Nano-vLLM 的建议顺序

1. **先看入口**：[example.py](example.py) → [llm.py](nanovllm/llm.py) → [llm_engine.py](nanovllm/engine/llm_engine.py) 了解整体流程
2. **理解抽象**：[sequence.py](nanovllm/engine/sequence.py) + [config.py](nanovllm/config.py) 搞清楚数据结构
3. **核心调度**：[scheduler.py](nanovllm/engine/scheduler.py) + [block_manager.py](nanovllm/engine/block_manager.py) 这是最值得深读的部分
4. **模型执行**：[model_runner.py](nanovllm/engine/model_runner.py) 理解 prefill/decode 的数据准备和 CUDA Graph
5. **模型层**：[qwen3.py](nanovllm/models/qwen3.py) + [attention.py](nanovllm/layers/attention.py) + [linear.py](nanovllm/layers/linear.py) 理解 TP 和 Flash Attention
6. **工具层**：[context.py](nanovllm/utils/context.py) + [loader.py](nanovllm/utils/loader.py)

### 关键技术之间的关系

```
显存管理 (PagedAttention)
    │
    ├── 使能 ──→ Prefix Caching (共享相同前缀的 block)
    │
    ├── 使能 ──→ Continuous Batching (动态分配/释放 block)
    │
    └── 使能 ──→ Preemption (换出 block 腾空间)

调度策略
    │
    ├── Chunked Prefill (大 prefill 分块，减少 TTFT)
    │
    └── Prefill/Decode 分离调度

执行优化
    │
    ├── CUDA Graph (消除 decode kernel launch overhead)
    │
    ├── Tensor Parallelism (多 GPU 并行)
    │
    ├── Flash Attention (高效 attention 实现)
    │
    └── torch.compile (算子融合)
```

### 推荐阅读顺序

1. 阅读本教程获取全局视角
2. 对照源码按上述学习顺序过一遍
3. 自行运行 `example.py`，加断点/日志观察每个 step 的调度和执行
4. 尝试修改：比如添加 Top-P 采样、支持新的模型架构
5. 阅读 vLLM 原论文和源码，理解生产级实现的更多细节

---

祝你学习愉快，期待看到你的推理框架开源项目！
