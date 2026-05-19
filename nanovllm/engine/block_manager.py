"""
BlockManager — KV Cache 显存管理器 + Prefix Caching
=====================================================
这是 Nano-vLLM 最核心、最精妙的部分。

核心思想来自 vLLM 的 PagedAttention:
  像操作系统管理虚拟内存一样管理 KV Cache
  - KV Cache 被切分为固定大小的 Block（类似内存页，默认 256 token/block）
  - 每个 Sequence 维护一个 block_table（类似页表），映射逻辑位置 → 物理 block
  - 多个 Sequence 可以共享同一个物理 block（Prefix Caching 的基础）

Prefix Caching 算法:
  对于每个 block 的 token 内容计算链式哈希:
    hash(block_0) = xxhash64(tokens_0)
    hash(block_i) = xxhash64(tokens_i, hash(block_{i-1}))
  链式哈希保证了: 相同前缀 → 相同哈希链 → 可安全复用 KV Cache.

  新请求到来时，从前往后匹配:
    请求 B 的 block_0 哈希 = H1 → hash_to_block_id[H1] = block_3 → 命中!
    请求 B 的 block_1 哈希 = H2 → hash_to_block_id[H2] = block_7 → 命中!
    请求 B 的 block_2 哈希 = H3 → 没找到 → 需要为新 block 计算 KV Cache
  如果命中的 block 仍在显存中（used_block_ids），直接复用，ref_count++.
  如果命中的 block 已被释放（不在 used_block_ids 中），重新加载到显存（ref_count=1）.
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    一个 KV Cache 物理块。

    每个 block 存储 block_size 个 token 的 Key 和 Value（在 GPU 显存中）。
    这里的 Block 对象只维护元数据，真正的 K/V 数据在 GPU tensor 中。

    字段:
        block_id:   物理块编号 (0 ~ num_blocks-1)
        ref_count:  引用计数（有多少个 Sequence 在共享这个 block）
        hash:       该 block 内容的 xxhash64 哈希值（用于 prefix cache 匹配）
        token_ids:  该 block 包含的 token 列表（用于验证哈希冲突）
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1         # -1 = 未计算哈希
        self.token_ids = []    # 存储 token_ids 是为了在哈希命中时做二次验证

    def update(self, hash: int, token_ids: list[int]):
        """注册哈希和 token（在 block 完成计算后调用）"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置 block 状态（重新分配时调用）"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    KV Cache Block 管理器。

    核心数据结构:
        blocks: list[Block]
            所有物理 block，索引即物理地址
        hash_to_block_id: dict[int, int]
            哈希值 → block_id 的映射（prefix cache 的核心数据结构）
            注意: block 被释放后，这个映射保留，后续请求仍可能命中
        free_block_ids: deque[int]
            空闲 block 队列（FIFO）
        used_block_ids: set[int]
            当前在显存中的 block 集合
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # ========== 哈希计算 ==========

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一个 block 的链式哈希。

        参数:
            token_ids: 当前 block 的 token 列表
            prefix:    前一个 block 的哈希值（-1 表示没有前缀）

        链式哈希: hash(block_i) = xxhash64(token_ids_i, hash(block_{i-1}))
        这保证了:
          1. 相同前缀产生相同的哈希链
          2. 不同前缀一定产生不同的哈希链（碰撞概率极低）
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))  # 先混入前一个 hash
        h.update(np.array(token_ids).tobytes())     # 再混入当前 token
        return h.intdigest()

    # ========== 底层 block 分配/释放 ==========

    def _allocate_block(self) -> int:
        """从空闲队列取一个 block"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, "分配的 block 应该未被引用"

        # 如果该 block 还有旧哈希（之前被使用过），清理映射
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        block.reset()                          # 重置为 ref_count=1
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """归还 block 到空闲队列（但不清理哈希映射，后续请求仍可命中）"""
        assert self.blocks[block_id].ref_count == 0, "释放时引用计数必须为 0"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        # 注意: hash_to_block_id 中的映射保留！
        # 这是 prefix cache 跨请求复用的关键：
        # block 虽然被释放（显存可被覆写），但哈希映射还在。
        # 后续请求如果命中该哈希，会重新将 block 加载到显存。

    # ========== Prefix Caching 核心算法 ==========

    def can_allocate(self, seq: Sequence) -> int:
        """
        检查能否为 seq 分配 KV Cache，同时计算 prefix cache 命中情况。

        从前往后遍历 seq 的每个 block:
          1. 计算链式哈希（混入前一个 block 的哈希）
          2. 在 hash_to_block_id 中查找
          3. 命中且 token_ids 匹配 → 缓存命中
          4. 未命中或 token_ids 不匹配 → 停止（后续 block 必然不命中）

        返回:
            -1:  显存不足，无法分配
            >=0: 命中的 block 数（这些 block 的 KV Cache 可以复用）
        """
        h = -1
        num_cached_blocks = 0       # 前缀命中的 block 数
        num_new_blocks = seq.num_blocks  # 需要新分配的 block 数

        for i in range(seq.num_blocks - 1):  # 最后一个 block 不参与 prefix cache（未满）
            token_ids = seq.block(i)

            # 链式哈希: 当前 token_ids + 前一个 hash
            h = self.compute_hash(token_ids, h)

            # 查找是否命中
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 未命中或哈希冲突（token_ids 不相等），停止匹配
                break

            num_cached_blocks += 1

            # 如果命中的 block 已在显存中（被其他 seq 引用），不需要新分配
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        # 检查剩余空闲 block 是否够用
        if len(self.free_block_ids) < num_new_blocks:
            return -1  # 显存不足

        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        为 Sequence 分配 block_table。

        分为两步:
          1. 复用缓存的 block（num_cached_blocks 个）: ref_count++ 或重新加载
          2. 分配新 block（剩余部分）: 从 free 队列取
        """
        assert not seq.block_table, "seq 不应该已有 block_table"

        # —— 第一步：复用前缀缓存 ——
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]   # can_allocate 已确认命中
            block = self.blocks[block_id]

            if block_id in self.used_block_ids:
                # block 在显存中 → 共享，引用计数 +1
                block.ref_count += 1
            else:
                # block 已被释放但哈希还在 → 重新分配（从 free 队列取出）
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)

            seq.block_table.append(block_id)

        # —— 第二步：分配新 block ——
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        # 更新已缓存的 token 数
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """
        释放 Sequence 的 block_table。

        对每个 block ref_count--，减到 0 的归还 free 队列。
        注意: hash_to_block_id 中的映射保留，后续请求仍可能命中。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # ========== Decode 阶段的 block 追加 ==========

    def can_append(self, seq: Sequence) -> bool:
        """
        检查 decode 阶段是否需要新 block。

        decode 时序列每次增长 1 个 token。当 token 数正好越过 block 边界时
        （即 len(seq) % block_size == 1，表示刚进入新 block 的第一个 token），
        需要分配一个额外的 block。

        返回: True = 不需要新 block 或可以分配，False = 需要新 block 但显存不足
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        如果序列长度刚好越过 block 边界（需要新 block），分配之。
        """
        if len(seq) % self.block_size == 1:
            # 刚进入新 block 的第一个 token → 需要新 block
            seq.block_table.append(self._allocate_block())

    # ========== 注册已完成 block 的哈希 ==========

    def hash_blocks(self, seq: Sequence):
        """
        在 postprocess 中调用。对已完成计算的 block 计算哈希并注册。

        只在 block 完全填满时才注册（因为未满的 block 内容还不完整，
        不能被其他序列复用）。

        具体: 从 (num_cached_tokens/block_size) 到
              ((num_cached_tokens+num_scheduled_tokens)/block_size)
              之间的完整 block 都会被哈希注册。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return  # 没有新的完整 block 需要注册

        # 链式哈希：从前一个 block 的哈希开始
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1

        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)        # 链式哈希
            block.update(h, token_ids)                  # 记录到 block 元数据
            self.hash_to_block_id[h] = block.block_id   # 注册到全局哈希表
