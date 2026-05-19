"""
LLMEngine — Nano-vLLM 的推理引擎主循环
=========================================
这是整个推理框架的"大脑"，负责：
1. 接收用户请求 (prompts)
2. 循环调用 Scheduler（调度） + ModelRunner（执行）
3. 直到所有请求完成，返回结果

调用链: 用户 → LLM.generate() → LLMEngine.generate() → step() × N → 返回文本

整体架构:
  LLMEngine
  ├── Scheduler     — 决定"这轮跑哪些序列、跑 prefill 还是 decode"
  ├── BlockManager  — 管理 KV Cache 显存分配（Scheduler 内部使用）
  └── ModelRunner   — 执行模型前向计算（可能跨多 GPU）
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """
        参数:
            model: 模型路径 (本地 HuggingFace 目录)
            **kwargs: 可选的 Config 参数，例如:
                - tensor_parallel_size: 张量并行 GPU 数量 (默认 1)
                - max_num_batched_tokens: 每轮最多处理多少 token (默认 16384)
                - max_num_seqs: 同时处理的最大序列数 (默认 512)
                - gpu_memory_utilization: GPU 显存利用率 (默认 0.9)
                - enforce_eager: 禁用 CUDA Graph (默认 False，调试时可设为 True)
        """
        # —————— 1. 分离出 Config 中定义的参数 ——————
        # 用户可能传了额外参数（如 use_tqdm），只提取 Config 中存在的字段
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # —————— 2. 设置 Sequence 的类变量 ——————
        # block_size 决定每个 KV Cache block 容纳多少 token（通常 256）
        # 所有 Sequence 实例共享此值
        Sequence.block_size = config.kvcache_block_size

        # —————— 3. 启动 Tensor Parallelism 的工作进程 (rank 1~N-1) ——————
        # 如果只用单卡 (tensor_parallel_size=1)，这个循环不执行
        # 多卡时，rank=0 是主进程，rank=1~N-1 是子进程
        # 子进程在 ModelRunner.__init__ 末尾进入 loop() 无限等待主进程指令
        self.ps = []        # 子进程列表 (Process 对象)
        self.events = []    # 跨进程 Event 信号，用于通知子进程"有新任务"
        ctx = mp.get_context("spawn")  # Windows/Mac 必须用 spawn，Linux 可用 fork
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()                                        # 创建 Event 信号
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()                                            # 启动子进程
            self.ps.append(process)
            self.events.append(event)

        # —————— 4. 创建主进程的 ModelRunner (rank=0) ——————
        # rank=0 负责：调度决策、数据准备、采样、通过 shared memory 指挥其他 rank
        # 传入 self.events 让 rank0 可以通知所有子进程
        self.model_runner = ModelRunner(config, 0, self.events)

        # —————— 5. 加载分词器 ——————
        # use_fast=True: 使用 Rust 实现的快速 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id  # 记录 EOS token ID，用于判断序列结束

        # —————— 6. 创建调度器 ——————
        # Scheduler 内部持有 BlockManager（管理 KV Cache 显存）
        self.scheduler = Scheduler(config)

        # —————— 7. 注册进程退出时的清理回调 ——————
        # 确保无论如何退出都能正确释放 CUDA 资源和子进程
        atexit.register(self.exit)

    def exit(self):
        """进程退出时清理：通知所有 rank 退出，回收子进程"""
        self.model_runner.call("exit")   # rank0 通过 shared memory 通知所有子进程退出
        del self.model_runner            # 触发 ModelRunner 析构 → 释放 CUDA 资源
        for p in self.ps:
            p.join()                     # 等待每个子进程结束

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        将一个请求转化为 Sequence 并加入调度队列。

        参数:
            prompt: 可以是字符串（会自动 tokenize）或已编码的 token_id 列表
            sampling_params: 采样参数
        """
        if isinstance(prompt, str):
            # 字符串 → token_id 列表（使用 tokenizer 编码）
            prompt = self.tokenizer.encode(prompt)
        # 创建 Sequence 对象，初始状态为 WAITING
        seq = Sequence(prompt, sampling_params)
        # 加入调度器的 waiting 队列
        self.scheduler.add(seq)

    def step(self):
        """
        执行一轮 **调度 → 执行 → 后处理**。

        这是引擎的核心循环体，每一步做三件事：

        ① Scheduler.schedule()
           从 waiting/running 队列中选出本轮要处理的序列
           决定是 prefill 阶段还是 decode 阶段
           为序列分配/扩展 KV Cache block_table

        ② ModelRunner.run()
           准备输入数据（input_ids, positions, slot_mapping 等）
           执行模型前向计算 → logits
           Sampler 采样 → 各序列的下一个 token_id

        ③ Scheduler.postprocess()
           将完成的 block 的哈希注册到 prefix cache
           更新序列状态（追加 token, 更新 num_cached_tokens）
           标记完成的序列为 FINISHED，释放 KV Cache

        返回:
            outputs: list[(seq_id, completion_token_ids)]
                    本轮完成的序列列表（可能为空）
            num_tokens: int
                    本轮处理的 token 数
                    prefill 时为正（= 处理的 prompt token 数）
                    decode 时为负（= -序列数，用于吞吐量计算的区分）
        """
        # —— 第 1 步：调度 ——
        seqs, is_prefill = self.scheduler.schedule()

        # 计算本轮处理的 token 数
        # prefill: 各 seq 的 num_scheduled_tokens 之和（每个 seq 可能 >1）
        # decode:  每个 seq 都只处理 1 个 token，所以用 -len(seqs)，负号用于区分阶段
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        # —— 第 2 步：模型执行 ——
        # ModelRunner.run() 完成：
        #   a) prepare_prefill() 或 prepare_decode() 准备输入张量
        #   b) set_context() 设置全局 Context（各层通过 get_context() 读取）
        #   c) model.forward() → hidden_states → compute_logits() → logits
        #   d) sampler() → 各序列的下一个 token_id
        # 返回 token_ids 列表，与 seqs 一一对应
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # —— 第 3 步：后处理 ——
        # a) hash_blocks(): 对已完成的 block 计算哈希，注册到 hash_to_block_id
        # b) 更新 seq.num_cached_tokens，追加 token_id 到 seq.token_ids
        # c) 判断是否结束（遇到 EOS 或达到 max_tokens），结束则释放 KV Cache
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 收集本轮已完成的序列
        # 注意：这里的 "完成" 是 postprocess 中刚标记的 FINISHED
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已完成（waiting 和 running 队列均为空）"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        完整的生成流程 —— 用户调用的唯一入口。

        参数:
            prompts: 输入提示列表，每个元素可以是:
                     - str:  文本（引擎内自动 tokenize）
                     - list[int]: 已编码的 token_id 列表
            sampling_params:
                     单个 SamplingParams → 所有 prompt 共用
                     list[SamplingParams] → 每个 prompt 独立配置
            use_tqdm: 是否显示进度条（benchmark 时可关闭）

        返回:
            list[dict]: 每个元素格式为:
                {"text": "解码后的生成文本", "token_ids": [生成的 token_id 列表]}

        工作流程:
            1. 所有 prompt 转为 Sequence，加入 waiting 队列
            2. 循环调用 step()，每轮处理一批 token
            3. 进度条实时显示 prefill/decode 吞吐量
            4. 所有序列完成后，按输入顺序返回结果
        """
        # —————— 1. 初始化进度条 ——————
        # total=len(prompts): 进度按完成请求数推进，每次有一个 seq 完成就 +1
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # —————— 2. 统一参数格式 ——————
        # 如果传了一个 SamplingParams，复制 N 份给每个 prompt
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # —————— 3. 将所有请求加入调度队列 ——————
        # 每个 prompt 创建一个 Sequence，状态 = WAITING，放入 waiting 队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # —————— 4. 主循环：不断 step() 直到所有请求完成 ——————
        outputs = {}        # {seq_id: completion_token_ids} 收集已完成序列的结果
        prefill_throughput = decode_throughput = 0.

        while not self.is_finished():
            t = perf_counter()                         # 开始计时（用于吞吐量计算）

            output, num_tokens = self.step()           # 执行一轮

            # 计算本轮吞吐量
            # num_tokens > 0 → prefill: num_tokens = 处理的 prompt token 数
            # num_tokens < 0 → decode:  -num_tokens = 生成的 token 数（每个 seq 1 个）
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)

            # 进度条显示实时吞吐量
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 收集本轮完成的序列，推进进度条
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)                         # 进度 +1

        pbar.close()

        # —————— 5. 按输入顺序整理结果 ——————
        # 按 seq_id 排序（seq_id 是递增分配的，所以排序 = 恢复输入顺序）
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # token_ids → 文本 + 保留原始 token_ids
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        return outputs
