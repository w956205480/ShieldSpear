import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import os

# 导入所需的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoder.params_model import *
from encoder.params_data import *
from ResNet.encoder_adapter import ResNetSpeakerEncoder, load_resnet_encoder

# 全局变量，用于存储模型和设备信息
_model = None
_device = None

# 嵌入向量的维度
EMBEDDING_SIZE = model_embedding_size

def load_model(weights_fpath: Path, device=None):
    """
    加载ResNet编码器模型的权重，并将模型设置为评估模式。
    
    参数:
        weights_fpath: 模型权重的路径。
        device: 设备（例如 "cpu", "cuda"）。如果为None，将自动选择可用的设备。
    """
    global _model, _device
    
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    else:
        _device = device
    
    print(f"尝试加载模型权重: {weights_fpath}")
    
    # 直接创建模型实例
    _model = ResNetSpeakerEncoder(_device)
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(weights_fpath, map_location=_device)
        
        # 判断checkpoint类型
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                try:
                    # 尝试加载model_state
                    _model.load_state_dict(checkpoint["model_state"])
                    print(f"成功加载ResNet模型。训练步骤: {checkpoint.get('step', '未知')}")
                except Exception as e:
                    print(f"警告: 加载模型状态时出错: {e}")
                    print("尝试部分加载模型参数...")
                    
                    # 部分加载参数
                    model_dict = _model.state_dict()
                    pretrained_dict = checkpoint["model_state"]
                    
                    # 筛选出大小匹配的参数
                    filtered_dict = {}
                    for k, v in pretrained_dict.items():
                        if k in model_dict and v.size() == model_dict[k].size():
                            filtered_dict[k] = v
                    
                    # 更新模型
                    model_dict.update(filtered_dict)
                    _model.load_state_dict(model_dict, strict=False)
                    print(f"部分加载完成。成功加载了 {len(filtered_dict)}/{len(model_dict)} 个参数。")
            else:
                # 尝试直接加载整个字典作为状态字典
                try:
                    _model.load_state_dict(checkpoint)
                    print("成功加载模型状态字典")
                except Exception as e:
                    print(f"警告: 加载状态字典时出错: {e}")
                    print("尝试部分加载模型参数...")
                    _model.load_state_dict(checkpoint, strict=False)
                    print("部分加载完成（忽略了一些参数）")
        else:
            # 尝试直接加载
            try:
                _model.load_state_dict(checkpoint)
                print("成功加载模型")
            except Exception as e:
                print(f"警告: 加载模型时出错: {e}")
                print("无法加载模型，将使用随机初始化的权重")
    except Exception as e:
        print(f"加载模型文件时出错: {e}")
        print("将使用随机初始化的权重")
    
    # 设置为评估模式
    _model.eval()
    print("ResNet模型已设置为评估模式")

def is_loaded():
    """
    检查模型是否已加载
    
    返回:
        如果模型已加载，则为True，否则为False
    """
    return _model is not None

def embed_mel_spectrogram(mel_spectrogram):
    """
    从梅尔频谱图提取嵌入向量
    
    参数:
        mel_spectrogram: 梅尔频谱图，形状为(n_frames, n_mels)
        
    返回:
        嵌入向量，形状为(EMBEDDING_SIZE,)
    """
    if _model is None:
        raise Exception("模型未加载。请先调用load_model()。")
    
    # 确保梅尔频谱图是浮点数
    mel_spectrogram = mel_spectrogram.astype(np.float32)
    
    # 转换为torch张量
    # 模型期望输入形状为(batch_size, n_frames, n_channels)
    x = torch.from_numpy(mel_spectrogram).float().unsqueeze(0)  # 添加批次维度
    
    # 移动到正确的设备
    x = x.to(_device)
    
    # 前向传播
    with torch.no_grad():
        y = _model(x).cpu()
    
    return y.squeeze().numpy()

def embed_utterance(wav_file_path=None, mel_spectrogram=None):
    """
    从音频文件或预处理的梅尔频谱图中提取说话人嵌入向量
    
    参数:
        wav_file_path: 音频文件的路径（可选）
        mel_spectrogram: 预处理的梅尔频谱图（可选）
        
    返回:
        说话人嵌入向量，形状为(EMBEDDING_SIZE,)
    """
    if _model is None:
        raise Exception("模型未加载。请先调用load_model()。")
    
    if mel_spectrogram is not None:
        # 直接使用提供的梅尔频谱图
        return embed_mel_spectrogram(mel_spectrogram)
    elif wav_file_path is not None:
        # 从音频文件加载梅尔频谱图
        # 注意：这需要音频处理库，如librosa
        try:
            import librosa
            
            # 加载音频文件
            y, sr = librosa.load(wav_file_path, sr=16000)  # 确保采样率为16kHz
            
            # 提取梅尔频谱图
            # 参数与SV2TTS保持一致
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_fft=int(0.025 * sr),  # 25ms窗口
                hop_length=int(0.01 * sr),  # 10ms步长
                n_mels=40  # 40个梅尔频带
            ).T  # 转置为(n_frames, n_mels)
            
            return embed_mel_spectrogram(mel_spectrogram)
        except ImportError:
            print("缺少librosa库。请安装它以处理音频文件: pip install librosa")
            raise
        except Exception as e:
            print(f"处理音频文件时出错: {e}")
            raise
    else:
        raise ValueError("必须提供wav_file_path或mel_spectrogram参数之一")

def load_mel_from_file(npy_file):
    """
    从.npy文件加载预处理好的梅尔频谱图
    
    参数:
        npy_file: .npy文件的路径
        
    返回:
        梅尔频谱图
    """
    try:
        mel_spectrogram = np.load(npy_file)
        return mel_spectrogram
    except Exception as e:
        print(f"加载梅尔频谱图文件时出错: {e}")
        raise

# 示例用法
if __name__ == "__main__":
    # 加载模型
    model_path = Path("./saved_models/resnet_encoder.pt")
    load_model(model_path)
    
    # 从.npy文件加载梅尔频谱图（如果有的话）
    try:
        mel_path = Path("./path/to/mel_spectrogram.npy")
        mel = load_mel_from_file(mel_path)
        
        # 提取嵌入向量
        embedding = embed_mel_spectrogram(mel)
        print(f"嵌入向量形状: {embedding.shape}")
    except:
        print("没有找到测试用的梅尔频谱图文件，跳过测试。")