from ResNet.encoder_adapter import ResNetSpeakerEncoder
from encoder.params_data import *
from encoder.audio import preprocess_wav, wav_to_mel_spectrogram  # 保持音频处理一致性
from pathlib import Path
import numpy as np
import torch

_model = None  # type: ResNetSpeakerEncoder
_device = None  # type: torch.device

# 导入采样率常量，以便与原始encoder保持一致
sampling_rate = 16000  # 与encoder/audio.py中定义的采样率保持一致

# 直接导出preprocess_wav函数，保持与原始encoder的一致性
# 通过这种方式，对于调用者来说，两个encoder模块的接口是一致的


def load_model(weights_fpath: Path, device=None):
    """
    加载ResNet编码器模型到内存中。如果此函数未被显式调用，将在首次调用embed_frames_batch()时
    使用默认权重文件运行。

    :param weights_fpath: 保存的模型权重路径。
    :param device: 设备名称或torch设备（例如"cpu"、"cuda"）。模型将被加载并在此设备上运行。
    输出将始终在CPU上。如果为None，将默认使用GPU（如果可用），否则使用CPU。
    """
    global _model, _device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    
    print(f"加载ResNet编码器模型: {weights_fpath}")
    
    # 初始化模型
    _model = ResNetSpeakerEncoder(_device, torch.device("cpu"))
    
    try:
        # 加载权重
        checkpoint = torch.load(weights_fpath, map_location=_device)
        
        # 检查checkpoint格式
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                try:
                    # 尝试加载model_state
                    _model.load_state_dict(checkpoint["model_state"])
                    print(f"成功加载模型。训练步骤: {checkpoint.get('step', '未知')}")
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
    return _model is not None


def embed_frames_batch(frames_batch):
    """
    为一批梅尔频谱图计算嵌入向量。

    :param frames_batch: 梅尔频谱图批次，形状为(batch_size, n_frames, n_channels)的numpy数组
    :return: 嵌入向量，形状为(batch_size, model_embedding_size)的numpy数组
    """
    if _model is None:
        raise Exception("模型未加载。在进行推理前调用load_model()。")

    frames = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed


def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    计算如何分割语音波形及其对应的梅尔频谱图，以获得每个<partial_utterance_n_frames>帧的部分语音。
    返回波形和梅尔频谱图切片，使每个部分语音波形对应其频谱图。此函数假定使用的梅尔频谱图参数是
    params_data.py中定义的参数。

    返回的范围可能会超出波形的长度。建议使用零填充波形直到wave_slices[-1].stop。

    :param n_samples: 波形中的样本数
    :param partial_utterance_n_frames: 每个部分语音中的梅尔频谱图帧数
    :param min_pad_coverage: 当达到最后一个部分语音时，它可能有足够的帧也可能没有。如果至少有
    <min_pad_coverage>的<partial_utterance_n_frames>存在，则最后一个部分语音将被考虑，就像
    我们填充了音频一样。否则，它将被丢弃，就像我们修剪了音频一样。如果没有足够的帧用于1个部分语音，
    则忽略此参数，以便函数始终至少返回1个切片。
    :param overlap: 部分语音应重叠的程度。如果设置为0，则部分语音完全不相交。
    :return: 波形切片和梅尔频谱图切片，作为数组切片的列表。分别使用这些切片索引波形和梅尔频谱图，
    以获得部分语音。
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # 计算切片
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # 评估是否值得额外填充
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
    """
    为单个语音计算嵌入向量。

    :param wav: 预处理的（见audio.py）语音波形，作为float32的numpy数组
    :param using_partials: 如果为True，则语音被分割为<partial_utterance_n_frames>帧的部分语音，
    并且语音嵌入是从它们的归一化平均值计算的。如果为False，则通过将整个频谱图馈送到网络来计算语音。
    :param return_partials: 如果为True，部分嵌入也将与对应于部分嵌入的wav切片一起返回。
    :param kwargs: compute_partial_splits()的附加参数
    :return: 嵌入向量，作为形状为(model_embedding_size,)的float32的numpy数组。如果<return_partials>
    为True，部分语音作为形状为(n_partials, model_embedding_size)的float32的numpy数组和作为
    切片列表的wav部分也将被返回。如果同时将<using_partials>设置为False，这两个值将为None。
    """
    # 如果不使用部分语音，则处理整个语音
    if not using_partials:
        frames = wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # 计算在哪里将语音分割为部分，并在必要时填充
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # 将语音分割为部分
    frames = wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)

    # 从部分嵌入计算语音嵌入
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    # 确保嵌入向量是一维的
    if len(embed.shape) > 1:
        print(f"警告: 嵌入向量维度为 {embed.shape}，需要一维向量")
        embed = embed.reshape(-1)
    
    # 确保嵌入向量是float32类型
    if embed.dtype != np.float32:
        embed = embed.astype(np.float32)
        
    # 检查是否有NaN或Inf值
    if np.isnan(embed).any() or np.isinf(embed).any():
        print("警告: 嵌入向量包含NaN或Inf值，将替换为0")
        embed = np.nan_to_num(embed)
        # 重新规范化
        norm = np.linalg.norm(embed, 2)
        if norm > 0:
            embed = embed / norm
        else:
            print("警告: 嵌入向量全为0，使用随机规范化向量")
            embed = np.random.randn(len(embed))
            embed = embed / np.linalg.norm(embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def embed_speaker(wavs, **kwargs):
    raise NotImplementedError()


def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    """
    将嵌入向量绘制为热图。这是一个辅助函数，用于可视化嵌入向量。
    
    :param embed: 要绘制的嵌入向量
    :param ax: matplotlib轴对象（可选）
    :param title: 图表标题
    :param shape: 热图形状（可选）
    :param color_range: 颜色范围
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    if ax is None:
        ax = plt.gca()

    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)

    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title) 