"""
Scheduler — 调度器
==================
决定每个 step 中"跑哪些序列、跑 prefill 还是 decode、分配多少 token budget"。

核心数据结构:
    waiting: deque[Sequence]  — 等待 prefill 的序列
    running: deque[Sequence]  — 正在逐 token 生成的序列

调度策略（两个阶段，prefill 优先）:
    阶段 1 — Prefill:
        从 waiting 队列逐个取出序列，为其分配 KV Cache block_table，
        确定本轮处理的 token 数（支持 chunked prefill）
        直到 token budget 用完、或显存不足、或达到最大 batch 数
        将完成 prefill 的序列移入 running 队列

    阶段 2 — Decode:
        当没有 prefill 可做时，从 running 队列循环取出序列
        每个序列只处理 1 个 token
        如果显存不足，抢占（preempt）末尾的序列，释放其 KV Cache

Preempt（抢占）机制:
    当 decode 阶段显存不足时，选择 running 末尾的序列"换出":
    释放其 KV Cache，将其标记为 WAITING，放回 waiting 队列头部
    下次轮到它时重新 prefill（因为 KV Cache 已丢失）
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                # 同时处理的最大序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 每轮 token budget
        self.eos = config.eos                                  # EOS token ID
        self.block_size = config.kvcache_block_size            # KV Cache block 大小
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()  # 等待 prefill 的序列
        self.running: deque[Sequence] = deque()  # 正在 decode 的序列

    def is_finished(self):
        """所有请求是否都已完成"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新序列加入等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度函数，决定本轮要处理哪些序列。

        返回:
            (scheduled_seqs, is_prefill)
            is_prefill=True  → prefill 阶段（处理 prompt token）
            is_prefill=False → decode 阶段（逐 token 生成）
        """
        scheduled_seqs = []       # 本轮调度的序列列表
        num_batched_tokens = 0    # 本轮已分配的 token budget

        # ============================================================
        #  阶段 1: Prefill — 从 waiting 队列取序列，处理它们的 prompt
        # ============================================================
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]  # 查看队首（不取出，因为可能被 chunked）

            # 计算剩余 token budget
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break  # token budget 用尽

            if not seq.block_table:
                # 首次调度该序列 → 检查能否分配 KV Cache
                # can_allocate 返回 prefix cache 命中的 block 数
                # 返回 -1 表示显存不足
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # 显存不足，等下一轮

                # 需要处理的 token 数 = 总 token - 已缓存的 token
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 之前被 chunked prefill 过，还有剩余 token 要处理
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # —— Chunked Prefill 的判断 ——
            # 如果 token budget 不够一次性完成该序列的 prefill:
            #   - 如果 scheduled_seqs 为空（它是本轮第一个 seq）→ 允许分块
            #   - 如果 scheduled_seqs 已有其他 seq → 不分块，留给下一轮
            # 这样设计的目的是: 优先保证简单情况（小 prompt）先完成
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                # 首次调度 → 正式分配 block_table
                self.block_manager.allocate(seq, num_cached_blocks)

            # 本轮处理 min(剩余 token, token_budget) 个 token
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 如果该序列的所有 prompt token 都已覆盖（包括缓存的 + 本轮处理的）
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # Prefill 完成 → 移入 running 队列准备 decode
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        # 如果有 prefill 序列被调度，直接返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # ============================================================
        #  阶段 2: Decode — 从 running 队列取序列，各生成 1 个 token
        # ============================================================
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()  # 从队首取出

            # 检查是否需要为这个 seq 分配新 block
            # decode 时序列每次 +1 token，可能刚好越过 block 边界
            while not self.block_manager.can_append(seq):
                # 显存不足，需要抢占（换出）
                if self.running:
                    # 抢占 running 队尾的序列（最"老"的序列）
                    self.preempt(self.running.pop())
                else:
                    # running 只剩当前这一个 seq，只能抢占它自己
                    self.preempt(seq)
                    break
            else:
                # can_append 成功 → 分配（如果需要的话）并加入调度
                seq.num_scheduled_tokens = 1    # decode 每次只处理 1 个 token
                seq.is_prefill = False          # 标记为 decode 阶段
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs, "调度结果不应为空"
        # 将调度的序列放回 running 队首（保持 FIFO 顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占（换出）一个序列。

        将其状态重置为 WAITING，释放 KV Cache，放回 waiting 队首。
        下次调度时会重新 prefill（因为 KV Cache 已被释放）。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True                    # 下次从头 prefill
        self.block_manager.deallocate(seq)       # 释放 KV Cache
        self.waiting.appendleft(seq)             # 放回队首，优先处理

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        模型执行后的后处理。

        对每个序列:
          1. 注册已完成 block 的哈希到 prefix cache
          2. 更新 num_cached_tokens
          3. 如果是 prefill 且未完成 → 继续 waiting
          4. 如果是 decode 或 prefill 完成 → 追加生成的 token
          5. 判断是否结束（EOS / max_tokens）
        """
        for seq, token_id in zip(seqs, token_ids):
            # —— 第 1 步：注册 block 哈希 ——
            # 将本步完成的完整 block 注册到全局哈希表，供后续请求复用
            self.block_manager.hash_blocks(seq)

            # —— 第 2 步：更新进度 ——
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # —— 第 3 步：prefill 未完成 → 继续等待 ——
            # Chunked prefill 的情况：该序列的 prompt 还没全处理完
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue  # 不追加 token，不检查结束，状态保持 RUNNING

            # —— 第 4 步：追加生成的 token ——
            seq.append_token(token_id)

            # —— 第 5 步：判断是否结束 ——
            # 条件 1: 遇到了 EOS token（且不忽略 EOS）
            # 条件 2: 生成的 token 数达到了 max_tokens 限制
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放 KV Cache
                self.running.remove(seq)            # 从运行队列移除
