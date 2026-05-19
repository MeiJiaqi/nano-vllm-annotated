"""
Linear — 支持 Tensor Parallelism 的线性层
============================================
实现了三种并行线性层，用于将模型切分到多个 GPU:

1. ColumnParallelLinear: 按列切分权重
   原始: Y = X @ W,   W: [in, out]
   TP:   Y_i = X @ W_i, W_i: [in, out/tp_size]
   每个 GPU 得到 Y 的一部分（列切分），后续计算各自独立
   用于: QKV 投影、gate_up 投影、Embedding

2. RowParallelLinear: 按行切分权重
   原始: Y = X @ W,   W: [in, out]
   TP:   Y_i = X_i @ W_i, W_i: [in/tp_size, out]
   各 GPU 的结果需要 all-reduce 求和
   用于: O 投影、down 投影

3. QKVParallelLinear: QKV 合并投影的列切分
   将 Wq, Wk, Wv 合并为一个大矩阵 [Wq | Wk | Wv]
   按 head 数切分，每个 GPU 持有各自的 Q heads + KV heads
   用于: Attention 的 QKV 投影

MergedColumnParallelLinear: 多输出合并的列切分
   将 gate_proj 和 up_proj 合并为 [W_gate | W_up]
   用于: MLP 的 gate+up 投影
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """整除检查"""
    assert numerator % denominator == 0
    return numerator // denominator


# ============================================================
#  LinearBase — 所有并行线性层的基类
# ============================================================
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,  # 切分的维度: 0=列切分, 1=行切分
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()    # 当前 GPU 的 rank
        self.tp_size = dist.get_world_size()  # GPU 总数

        # 权重参数（每个 GPU 只持有自己的一片）
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 自定义 weight_loader，用于从 safetensors 正确加载分片权重
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ============================================================
#  ReplicatedLinear — 普通线性层（不切分，所有 GPU 持完整权重）
# ============================================================
class ReplicatedLinear(LinearBase):

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接复制完整权重"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ============================================================
#  ColumnParallelLinear — 列切分线性层
# ============================================================
class ColumnParallelLinear(LinearBase):

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        # output_size 按列（dim 0）切分为 tp_size 份
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        加载权重：从完整权重中取当前 rank 对应的切片。

        例: output_size=1024, tp_size=4
          rank 0: rows [0:256]
          rank 1: rows [256:512]
          rank 2: rows [512:768]
          rank 3: rows [768:1024]
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向: Y_i = X @ W_i

        每个 GPU 得到输出的一部分列片。
        不需要 all-reduce（后续操作也在列片上独立完成）。
        """
        return F.linear(x, self.weight, self.bias)


# ============================================================
#  MergedColumnParallelLinear — 多输出合并的列切分
# ============================================================
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        """
        将多个线性层合并为一个矩阵乘法。

        例: gate_up_proj 合并 gate_proj(intermediate) + up_proj(intermediate)
            output_sizes = [intermediate, intermediate]
            总 output = 2 * intermediate
        """
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        加载合并权重：根据 loaded_shard_id 确定要写到哪一段。

        例: gate_up_proj, tp_size=2, intermediate=1024
          权重布局: [gate_0 | gate_1 | up_0 | up_1]（按 TP rank 交错）
                          ↑ loaded_shard_id=0  ↑ loaded_shard_id=1
        """
        param_data = param.data
        # 定位到当前 shard 在合并矩阵中的位置
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从 loaded_weight 中取当前 rank 的分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# ============================================================
#  QKVParallelLinear — QKV 合并投影的列切分
# ============================================================
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)        # 当前 rank 的 Q head 数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # 当前 rank 的 KV head 数

        # 输出维度: Q heads * head_dim + K heads * head_dim + V heads * head_dim
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        加载 Q、K、V 权重到合并矩阵的正确位置。

        合并权重布局: [Q_part | K_part | V_part]
          每个 part 内部按 TP rank 切分。

        参数:
            loaded_shard_id: "q", "k", 或 "v"
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]

        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0                                    # Q 在矩阵最前面
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size      # K 在 Q 之后
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads * self.head_size +    # V 在 Q 和 K 之后
                            self.num_kv_heads * self.head_size)

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# ============================================================
#  RowParallelLinear — 行切分线性层
# ============================================================
class RowParallelLinear(LinearBase):

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        # input_size 按行（dim 1）切分为 tp_size 份
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """加载权重：按行切分"""
        param_data = param.data
        if param_data.ndim == 1:  # bias 不需要切分
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向: Y_i = X_i @ W_i

        每个 GPU 计算部分结果，然后 all-reduce 求和获得完整输出。

        数学原理:
          Y = X @ W
            = [X_0 | X_1 | ...] @ [W_0]
                                  [W_1]
                                  [...]
            = X_0 @ W_0 + X_1 @ W_1 + ...
            = all_reduce(Y_i)
        """
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)  # 各 GPU 的结果求和
        return y
