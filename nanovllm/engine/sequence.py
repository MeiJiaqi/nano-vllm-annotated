"""
Sequence — 序列抽象
===================
代表一个完整的推理请求，包含:
- token 序列（prompt + 已生成部分）
- KV Cache 的 block_table（逻辑→物理映射）
- 状态机（WAITING → RUNNING → FINISHED）

类比: 操作系统中的"进程"，block_table 就像是进程的页表。
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列的三种状态"""
    WAITING = auto()   # 等待 prefill（在 waiting 队列中）
    RUNNING = auto()   # 正在生成（在 running 队列中）
    FINISHED = auto()  # 已完成（遇到 EOS 或达到 max_tokens）


class Sequence:
    """
    一个推理序列（请求）。

    核心字段:
        token_ids:      完整的 token 列表 = [prompt tokens] + [generated tokens]
        num_tokens:     当前总 token 数
        num_prompt_tokens: prompt 部分的 token 数（不变）
        num_cached_tokens: 已在 KV Cache 中的 token 数（通过 prefix cache 命中或已计算）
        num_scheduled_tokens: 本轮 step 计划处理的 token 数
        block_table:    KV Cache 的物理 block ID 列表（类似页表）
        is_prefill:     是否处于 prefill 阶段

    生命周期:
        WAITING ──(prefill 完成)──→ RUNNING ──(EOS/max_tokens)──→ FINISHED
           ↑                            │
           └────(被抢占 preempt)─────────┘
    """

    block_size = 256            # 类变量：每个 KV Cache block 容纳的 token 数
    counter = count()           # 类变量：自增 ID 分配器 (0, 1, 2, ...)

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # —— 唯一标识 ——
        self.seq_id = next(Sequence.counter)  # 自增 ID，用于排序输出和去重

        # —— 状态 ——
        self.status = SequenceStatus.WAITING  # 初始为等待状态

        # —— Token 数据 ——
        self.token_ids = copy(token_ids)       # 完整 token 列表（会随时间增长）
        self.last_token = token_ids[-1]        # 缓存最后一个 token（decode 时只需这个）
        self.num_tokens = len(self.token_ids)  # 当前总 token 数
        self.num_prompt_tokens = len(token_ids) # prompt 长度（固定不变）

        # —— KV Cache 相关 ——
        self.num_cached_tokens = 0             # 已在 KV Cache 中的 token 数
        self.num_scheduled_tokens = 0          # 本轮 step 要处理的 token 数
        self.is_prefill = True                 # 是否处于 prefill 阶段
        self.block_table = []                  # 物理 block ID 列表（页表）

        # —— 采样参数 ——
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # ========== 便捷属性 ==========

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        """支持切片和索引访问 token_ids"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数（不含 prompt）"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """prompt 部分的 token_ids"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """生成部分的 token_ids"""
        return self.token_ids[self.num_prompt_tokens:]

    # ========== Block 相关 ==========

    @property
    def num_blocks(self):
        """当前 token 序列需要多少个 block（向上取整）"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个 block 中有多少个 token（用于 decode 时计算 slot 位置）"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第 i 个 block 对应的 token 子序列"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """追加一个生成的 token"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # ========== 序列化（用于跨进程传输） ==========
    # 当 TP > 1 时，Sequence 需要通过 pickle 在进程间传输
    # 为了减少序列化开销，只传输必要字段

    def __getstate__(self):
        """pickle 序列化时调用：只传输核心状态"""
        # prefill 阶段需要完整 token_ids，decode 阶段只需要 last_token
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        )

    def __setstate__(self, state):
        """pickle 反序列化时调用：恢复状态"""
        (self.num_tokens, self.num_prompt_tokens,
         self.num_cached_tokens, self.num_scheduled_tokens,
         self.block_table, last_state) = state
        if isinstance(last_state, list):
            # prefill: 恢复完整 token_ids
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode: 只恢复 last_token，完整 token_ids 由主进程维护
            self.token_ids = []
            self.last_token = last_state
