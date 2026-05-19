"""
ModelRunner — 模型执行器
========================
负责将 Scheduler 的调度决策转化为实际的 GPU 计算。

这是连接"调度逻辑"和"模型权重"的桥梁:
  1. 准备 prefill/decode 输入数据（input_ids, positions, slot_mapping 等）
  2. 设置全局 Context（各层通过 get_context() 获取元信息）
  3. 运行模型 forward → logits → sampler → token_ids
  4. 管理 CUDA Graph（decode 加速）
  5. 管理 Tensor Parallelism 的跨进程通信

Prefill vs Decode 的数据差异:
  Prefill:  input_ids = [t0, t1, t2, ...]   (多个 token / seq)
            positions  = [0, 1, 2, ...]
            slot_mapping: 每个 token 映射到 KV Cache 物理位置
            使用 flash_attn_varlen_func (变长序列 attention)

  Decode:   input_ids = [last_token]         (只有 1 个 token / seq)
            positions  = [seq_len - 1]
            slot_mapping: 只有 1 个 slot (最后一个 block 的最后位置)
            使用 flash_attn_with_kvcache (带 KV Cache 的 attention)
            可以走 CUDA Graph 快速路径
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        参数:
            config: 全局配置
            rank:   当前进程的 GPU rank（0=主进程, 1+=子进程）
            event:  rank0 为空列表；子进程为单个 Event（用于等待主进程指令）
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager  # True=禁用 CUDA Graph
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # —— 步骤 1: 初始化 NCCL 进程组 ——
        # 所有 TP rank 通过 TCP 建立通信（单机多卡场景）
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # —— 步骤 2: 构建模型 ——
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)     # 切换到模型精度（如 bfloat16）
        torch.set_default_device("cuda")              # 默认在 GPU 上创建 tensor
        self.model = Qwen3ForCausalLM(hf_config)      # 构建模型结构
        load_model(self.model, config.model)          # 从 safetensors 加载权重
        self.sampler = Sampler()                      # 采样器

        # —— 步骤 3: 预热 ——
        # 跑一次 prefill 来预热 CUDA kernel、分配 workspace
        self.warmup_model()

        # —— 步骤 4: 分配 KV Cache ——
        # 根据可用显存计算能分配多少 block，创建 KV Cache 大 tensor
        self.allocate_kv_cache()

        # —— 步骤 5: 录制 CUDA Graph ——
        # 为 decode 阶段录制 CUDA Graph，消除 kernel launch overhead
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # —— 步骤 6: 多进程通信初始化 ——
        if self.world_size > 1:
            if rank == 0:
                # 主进程: 创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 等待子进程就绪
            else:
                # 子进程: 等待主进程创建共享内存后连接
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()     # 进入无限循环，等待主进程指令
                                # 注意: loop() 不会返回，直到收到 "exit"

    def exit(self):
        """清理资源"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  # 主进程删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # ========== 多进程通信 ==========

    def loop(self):
        """子进程的主循环：等待指令 → 执行 → 等待下一条"""
        while True:
            method_name, args = self.read_shm()      # 阻塞等待主进程指令
            self.call(method_name, *args)            # 执行
            if method_name == "exit":
                break

    def read_shm(self):
        """子进程: 从共享内存读取指令"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()                            # 等待主进程 set event
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化
        self.event.clear()                           # 清除信号
        return method_name, args

    def write_shm(self, method_name, *args):
        """主进程: 向共享内存写入指令"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 写入数据长度
        self.shm.buf[4:n+4] = data                    # 写入序列化数据
        for event in self.event:
            event.set()                               # 通知所有子进程

    def call(self, method_name, *args):
        """
        统一的 RPC 调用入口。

        主进程 (rank=0): 写入共享内存 → 通知子进程 → 本地执行
        子进程 (rank>0): 直接本地执行（由 loop 调用）
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ========== 初始化辅助 ==========

    def warmup_model(self):
        """预热 CUDA: 跑一次 prefill 触发所有 kernel 编译"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = \
            self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)           # 跑一次 prefill
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        计算并分配 KV Cache 显存。

        逻辑:
          total_available = total * gpu_memory_utilization  # GPU 总可用
          model_memory = peak - current                     # 模型权重的显存
          kv_cache_budget = total_available - model_memory  # 剩余给 KV Cache

        每个 block 的字节数:
          block_bytes = 2 × num_layers × block_size × num_kv_heads × head_dim × dtype_size
                       ↑ K + V

        可用 block 数 = kv_cache_budget / block_bytes
        """
        config = self.config
        hf_config = config.hf_config

        # 获取当前显存状态
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]    # warmup 期间的峰值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 计算单个 block 的字节数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)
        # 2× = K + V, num_layers, block_size tokens, kv_heads, head_dim, bytes_per_element
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                       num_kv_heads * head_dim * hf_config.dtype.itemsize)

        # 计算可分配的 block 数
        # total * utilization: GPU 显存可用总量
        # used - peak + current: 模型权重 + 固定开销
        # 差值 = 可分配给 KV Cache 的显存
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        assert config.num_kvcache_blocks > 0, "GPU 显存不足以分配任何 KV Cache block"

        # 创建 KV Cache 大 tensor
        # shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # dim 0: 0=K, 1=V
        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )

        # 将 KV Cache 切片绑定到每个 Attention 层
        # 每层直接持有自己那一层的 K/V cache 引用，避免索引开销
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # K cache for this layer
                module.v_cache = self.kv_cache[1, layer_id]  # V cache for this layer
                layer_id += 1

    # ========== 数据准备 ==========

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备 block_tables tensor。

        将各 seq 的 block_table 列表转为对齐的 2D tensor，
        不足的部分用 -1 填充。

        例: block_tables = [[3, 7, 2], [5, -1, -1]]
            表示 seq_0 用 3 个 block，seq_1 只用 1 个 block

        返回: shape = [num_seqs, max_num_blocks]
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32,
                                     pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 prefill 阶段的输入数据。

        核心工作:
          1. 为每个 seq 收集要处理的 token_ids 和 positions
          2. 计算 cu_seqlens（cumulative sequence lengths）用于 flash attention
          3. 计算 slot_mapping: 每个 token 在 KV Cache 中的物理位置
          4. 如果有 prefix cache（某些 seq 的 KV 比 Q 长），准备 block_tables

        返回: (input_ids, positions)
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]    # Q 的累积序列长度
        cu_seqlens_k = [0]    # K 的累积序列长度（prefix cache 命中时比 Q 长）
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []     # token_id → KV Cache 物理 slot
        block_tables = None

        for seq in seqs:
            start = seq.num_cached_tokens        # 从哪开始（跳过已缓存的）
            seqlen_q = seq.num_scheduled_tokens  # Q 的长度（本轮要处理的 token 数）
            end = start + seqlen_q
            seqlen_k = end                       # K 的长度（所有已计算的 token）

            # 收集 token_ids 和 positions
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))

            # 累积长度（flash attention varlen 需要）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:  # warmup 阶段，没有 block_table
                continue

            # 计算 slot_mapping
            # 每个 token 需要知道它应该存储在 KV Cache 的哪个物理位置
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # 物理 block 的基础 slot
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size  # 第一个 block 可能有偏移
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        # 如果有 prefix cache 命中 → K 比 Q 长 → 需要 block_tables 做间接寻址
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 转为 GPU tensor（pin_memory=True 加速 CPU→GPU 传输）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置全局 Context，各层通过 get_context() 读取
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 decode 阶段的输入数据。

        decode 阶段每个 seq 只处理最后 1 个 token:
          input_ids = [seq_0.last_token, seq_1.last_token, ...]
          positions  = [len(seq_0)-1, len(seq_1)-1, ...]

        slot_mapping 计算: 该 seq 最后一个 block 的最后一个位置
          slot = block_table[-1] * block_size + last_block_num_tokens - 1

        返回: (input_ids, positions)
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []   # 各序列当前的总长度（用于 flash_attn_with_kvcache）

        for seq in seqs:
            input_ids.append(seq.last_token)       # 只取最后一个 token
            positions.append(len(seq) - 1)         # 位置 = 当前长度 - 1
            context_lens.append(len(seq))          # 总长度

            # 该 token 应写入的 KV Cache slot
            # 例: block_size=256, last_block_num_tokens=50
            #     slot = block_table[-1]*256 + 49
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        # 转为 GPU tensor
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # 设置全局 Context
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数（各 seq 的 temperature 可能不同）"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32,
                                     pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ========== 模型执行 ==========

    @torch.inference_mode()  # 禁用 autograd，推理不需要梯度
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型 forward + compute_logits。

        三种执行路径:
          1. Prefill:           直接运行（输入长度不定，无法用 CUDA Graph）
          2. Decode (eager):    enforce_eager=True 时直接运行
          3. Decode (CUDA Graph): 用预录制的 graph replay
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 直接执行（prefill 或 batch 过大不适合 CUDA Graph）
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # —— CUDA Graph 快速路径 ——
            bs = input_ids.size(0)  # batch_size
            context = get_context()

            # 选择 ≥ bs 的最小预录制 graph
            # 例: bs=3 → 选 graph_bs=4 的 graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]

            graph_vars = self.graph_vars

            # 将真实数据填入预分配的 tensor 槽位
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)       # 多余的槽位填 -1（无效）
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放 CUDA Graph：一次 kernel launch 执行整个 forward
            graph.replay()

            # 只取有效 bs 的输出
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        完整的推理步骤。

        流程:
          1. 准备输入数据（prefill 或 decode）
          2. 准备采样参数
          3. 模型 forward → logits
          4. 采样 → token_ids
          5. 清除 Context

        返回: 各序列的下一个 token_id 列表（与 seqs 一一对应）
        """
        # 步骤 1: 准备输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill \
                               else self.prepare_decode(seqs)

        # 步骤 2: 采样参数（所有 rank 都需要，但只有 rank0 做采样）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 步骤 3: 模型前向
        logits = self.run_model(input_ids, positions, is_prefill)

        # 步骤 4: 采样（只有 rank0 做，子进程返回 None）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # 步骤 5: 清除全局 Context
        reset_context()
        return token_ids

    # ========== CUDA Graph 录制 ==========

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        为 decode 阶段录制 CUDA Graph。

        CUDA Graph 的原理:
          将一整个 GPU 操作序列（所有 kernel launch）录制为一个"图"。
          后续只需一次 replay 调用，GPU 就会按录制的顺序执行所有 kernel。
          这消除了 CPU 端逐 kernel 发 launch 指令的 overhead。

        限制:
          - 输入 tensor 的 shape 必须固定（因此需要为不同 batch_size 分别录制）
          - 只适用于 decode（每个 seq 固定 1 个 token，shape 可预测）
          - prefill 的输入长度可变，无法录制

        录制策略:
          为常见的 batch_size 分别录制 graph:
          graph_bs = [1, 2, 4, 8, 16, 32, 48, ...]
          运行时选 ≥ 实际 bs 的最小 graph
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 最大录制 batch_size
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 预分配 tensor（不同 batch_size 的 graph 共用这些 buffer）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # batch_size 列表: [1, 2, 4, 8, 16, 32, 48, 64, ...]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None  # CUDA Graph 内存池（共享）

        # 从大到小录制（大 batch 先创建内存池）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False,
                        slot_mapping=slot_mapping[:bs],
                        context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs])

            # 预热：先跑一次，确保 kernel 已编译
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # 录制 CUDA Graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            if self.graph_pool is None:
                self.graph_pool = graph.pool()  # 保存内存池，后续 graph 共享

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存所有 graph 共用的 tensor buffer
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
