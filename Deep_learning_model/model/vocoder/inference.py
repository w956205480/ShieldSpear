from vocoder.models.fatchord_version import WaveRNN
from vocoder import hparams as hp
import torch
import numpy as np
import librosa


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model, _device
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    
    try:
        checkpoint = torch.load(weights_fpath, _device)
        # 尝试加载不同格式的模型状态
        if 'model_state' in checkpoint:
            _model.load_state_dict(checkpoint['model_state'])
        else:
            _model.load_state_dict(checkpoint)
        print("Vocoder model loaded successfully")
    except Exception as e:
        print(f"Error loading vocoder model: {e}")
        print("Using uninitialized model weights")
    
    _model.eval()


def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=True, batched=True, target=8000, overlap=800, 
                   progress_callback=None):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        # 检查梅尔频谱图是否已经在正确的范围内
        mel_abs_max = np.abs(mel).max()
        if mel_abs_max > 2 * hp.mel_max_abs_value:  # 如果梅尔值异常大
            print(f"Warning: unusually large mel values detected (max={mel_abs_max}). "
                  f"Applying custom normalization.")
            mel = mel * (hp.mel_max_abs_value / mel_abs_max)
        
        # 标准归一化
        mel = mel / hp.mel_max_abs_value
    
    # 检查是否有NaN值
    if np.isnan(mel).any():
        print("Warning: NaN values found in mel spectrogram, replacing with zeros")
        mel = np.nan_to_num(mel)
    
    # 创建张量
    mel = torch.from_numpy(mel[None, ...])
    
    # 确保不超过vocoder能处理的最大长度(如果需要的话)
    max_mel_length = 12000  # 一个安全的上限值
    if mel.shape[2] > max_mel_length:
        print(f"Warning: mel spectrogram too long ({mel.shape[2]} frames), "
              f"truncating to {max_mel_length} frames")
        mel = mel[:, :, :max_mel_length]
    
    # 生成波形
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
    return wav


def apply_taper(wav, taper_length=2000):
    """
    对波形应用淡入淡出效果，以避免声音突然开始或结束
    
    :param wav: 输入的波形
    :param taper_length: 淡入淡出的样本数量
    :return: 应用了淡入淡出效果的波形
    """
    wav_length = len(wav)
    if wav_length <= taper_length * 2:
        # 如果波形长度太短，减小淡入淡出长度
        taper_length = wav_length // 4
    
    # 应用淡入
    fade_in = np.linspace(0.0, 1.0, taper_length)
    wav[:taper_length] *= fade_in
    
    # 应用淡出
    fade_out = np.linspace(1.0, 0.0, taper_length)
    wav[-taper_length:] *= fade_out
    
    return wav


def normalize_volume(wav, target_dBFS=-30):
    """
    将波形归一化到目标音量级别
    
    :param wav: 输入的波形
    :param target_dBFS: 目标音量分贝满刻度，通常是负值
    :return: 归一化后的波形
    """
    # 避免静音输入导致问题
    if np.abs(wav).max() < 1e-10:
        return wav
    
    # 计算当前dBFS
    mean_square = np.mean(wav ** 2)
    if mean_square > 0:
        current_dBFS = 10 * np.log10(mean_square)
        # 计算需要的增益
        gain = 10 ** ((target_dBFS - current_dBFS) / 20)
        # 应用增益，但避免过大的值
        gain = min(gain, 10.0)  # 限制最大增益为10倍
        wav = wav * gain
    
    # 确保在[-1, 1]范围内
    wav = np.clip(wav, -1.0, 1.0)
    
    return wav


# 声码器的采样率
sample_rate = hp.sample_rate
