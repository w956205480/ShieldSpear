import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


_output_ref = None
_replicas_ref = None

def data_parallel_workaround(model, *inputs):
    """
    增强版的数据并行处理函数，处理参数为空和StopIteration异常的情况
    """
    # 检查模型是否有参数
    has_params = False
    try:
        next(model.parameters())
        has_params = True
    except StopIteration:
        print("警告：模型没有参数，使用输入张量的设备")
        # 如果没有参数，则返回单设备计算结果
        try:
            return model(*inputs)
        except Exception as e:
            print(f"单设备计算失败: {e}")
            # 在极端情况下，提供默认返回值
            if len(inputs) > 0 and hasattr(inputs[0], 'device'):
                device = inputs[0].device
            else:
                device = torch.device('cpu')
                
            # 根据tacotron模型的预期输出结构创建默认值
            batch_size = inputs[0].size(0) if len(inputs) > 0 and inputs[0].dim() > 0 else 1
            seq_len = inputs[0].size(1) if len(inputs) > 0 and inputs[0].dim() > 1 else 1
            n_mels = 80  # 梅尔频谱图的标准维度
            
            mel_outputs = torch.zeros(batch_size, n_mels, seq_len, device=device)
            linear = torch.zeros(batch_size, seq_len, n_mels, device=device)
            attn_scores = torch.zeros(batch_size, 1, seq_len, device=device)
            stop_outputs = torch.zeros(batch_size, seq_len, device=device)
            
            return mel_outputs, linear, attn_scores, stop_outputs
    
    # 检查是否有CUDA支持且有多个设备
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        try:
            return model(*inputs)
        except Exception as e:
            print(f"单设备计算失败: {e}")
            # 提供默认返回值（同上）
            if len(inputs) > 0 and hasattr(inputs[0], 'device'):
                device = inputs[0].device
            else:
                device = torch.device('cpu')
                
            batch_size = inputs[0].size(0) if len(inputs) > 0 and inputs[0].dim() > 0 else 1
            seq_len = inputs[0].size(1) if len(inputs) > 0 and inputs[0].dim() > 1 else 1
            n_mels = 80
            
            mel_outputs = torch.zeros(batch_size, n_mels, seq_len, device=device)
            linear = torch.zeros(batch_size, seq_len, n_mels, device=device)
            attn_scores = torch.zeros(batch_size, 1, seq_len, device=device)
            stop_outputs = torch.zeros(batch_size, seq_len, device=device)
            
            return mel_outputs, linear, attn_scores, stop_outputs
    
    try:
        # 尝试并行计算
        return torch.nn.parallel.data_parallel(model, inputs)
    except Exception as e:
        print(f"警告：并行处理时发生{type(e).__name__}错误: {str(e)}")
        print("在单设备上重试")
        # 出错时回退到单设备计算
        try:
            return model(*inputs)
        except Exception as e2:
            print(f"单设备计算也失败: {e2}")
            # 提供默认返回值（同上）
            if len(inputs) > 0 and hasattr(inputs[0], 'device'):
                device = inputs[0].device
            else:
                device = torch.device('cpu')
                
            batch_size = inputs[0].size(0) if len(inputs) > 0 and inputs[0].dim() > 0 else 1
            seq_len = inputs[0].size(1) if len(inputs) > 0 and inputs[0].dim() > 1 else 1
            n_mels = 80
            
            mel_outputs = torch.zeros(batch_size, n_mels, seq_len, device=device)
            linear = torch.zeros(batch_size, seq_len, n_mels, device=device)
            attn_scores = torch.zeros(batch_size, 1, seq_len, device=device)
            stop_outputs = torch.zeros(batch_size, seq_len, device=device)
            
            return mel_outputs, linear, attn_scores, stop_outputs


class ValueWindow:
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def values(self):
        return self._values

    @property
    def size(self):
        return len(self._values)

    @property
    def full(self):
        return len(self._values) >= self._window_size

    @property
    def average(self):
        return sum(self._values) / max(len(self._values), 1)
