"""
Loader — 模型权重加载
=====================
从 HuggingFace 格式的 safetensors 文件加载权重。

packed_modules_mapping:
  模型定义中，QKV 投影和 Gate+Up 投影是合并的（一次矩阵乘法）。
  但 HuggingFace 的权重文件是按标准命名保存的（q_proj, k_proj, v_proj 分开）。
  packed_modules_mapping 记录了如何将分散的权重加载到合并后的层中。

  例:
    "q_proj" → ("qkv_proj", "q")
      表示: 将文件名中的 "q_proj" 替换为 "qkv_proj"，
           将权重加载到合并层中 "q" 对应的分片位置。
"""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载：直接复制"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    从 safetensors 文件加载模型权重。

    流程:
      1. 扫描 model 目录下的所有 .safetensors 文件
      2. 对每个权重名:
         a. 检查是否在 packed_modules_mapping 中
         b. 如果在 → 映射到合并后的参数名和分片位置
         c. 如果不在 → 直接按名称找参数
      3. 调用参数的 weight_loader() 方法加载权重
         （不同并行层有不同的 weight_loader，负责切分逻辑）
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:   # 在 CPU 上打开
            for weight_name in f.keys():           # 遍历所有权重名

                # 检查是否是合并权重（QKV / gate_up）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 例: "model.layers.0.self_attn.q_proj.weight"
                        #       → "model.layers.0.self_attn.qkv_proj.weight"
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        # 传入 shard_id 让 loader 知道写到哪个分片
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重（非合并），直接按名称加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
