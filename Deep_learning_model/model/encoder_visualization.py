#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import librosa.display

# 配置matplotlib支持中文显示
import matplotlib
# 使用以下方法之一设置中文字体
# 方法1：使用系统默认中文字体（推荐Windows用户）
matplotlib.rcParams['font.family'] = ['sans-serif']
# 方法2：明确指定中文字体（根据系统选择其中之一）
# Windows系统推荐字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong']
# Linux系统推荐字体
# matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Source Han Sans CN']
# macOS系统推荐字体
# matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Source Han Sans CN']
# 解决保存图像时负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

# 从项目导入所需模块
from encoder import inference as lstm_encoder
from ResNet import inference as resnet_encoder
from encoder.params_model import model_embedding_size
from utils.argutils import print_args


def preprocess_audio(audio_path):
    """处理音频文件，返回预处理后的音频数据"""
    # 尝试两种预处理方法都使用相同的实现
    waveform = lstm_encoder.preprocess_wav(audio_path)
    return waveform


def plot_embeddings_comparison(lstm_embed, resnet_embed, title=None, save_path=None):
    """将LSTM和ResNet编码器的嵌入向量并排可视化"""
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 计算嵌入向量的相似度
    similarity = np.dot(lstm_embed, resnet_embed)
    cosine_sim = similarity / (np.linalg.norm(lstm_embed) * np.linalg.norm(resnet_embed))
    
    # 设置图像显示大小
    height = int(np.sqrt(model_embedding_size))
    shape = (height, -1)
    
    # 绘制LSTM嵌入向量热图
    lstm_img = ax1.imshow(lstm_embed.reshape(shape), cmap='viridis')
    ax1.set_title(f"LSTM Encoder Embedding ({len(lstm_embed)} dim)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(lstm_img, ax=ax1, fraction=0.046, pad=0.04)
    
    # 绘制ResNet嵌入向量热图
    resnet_img = ax2.imshow(resnet_embed.reshape(shape), cmap='viridis')
    ax2.set_title(f"ResNet Encoder Embedding ({len(resnet_embed)} dim)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(resnet_img, ax=ax2, fraction=0.046, pad=0.04)
    
    # 添加总标题
    if title:
        fig.suptitle(f"{title}\nCosine Similarity: {cosine_sim:.4f}", fontsize=16)
    else:
        fig.suptitle(f"Encoder Embeddings Comparison\nCosine Similarity: {cosine_sim:.4f}", fontsize=16)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
        print(f"Embedding visualization saved to: {save_path}")
    
    plt.show()


def plot_mel_spectrograms(audio_path, save_path=None):
    """绘制原始音频的梅尔频谱图"""
    # 加载音频
    waveform = preprocess_audio(audio_path)
    
    # 计算梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=80, 
                                             hop_length=int(0.01 * 16000), 
                                             win_length=int(0.025 * 16000))
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 绘制梅尔频谱图
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=16000, hop_length=int(0.01 * 16000))
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram: {os.path.basename(audio_path)}")
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
        print(f"Mel spectrogram saved to: {save_path}")
    
    plt.show()


def process_audio_file(audio_path, lstm_model_path, resnet_model_path, output_dir, use_cpu=False):
    """处理单个音频文件并生成可视化"""
    # 设置设备
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载LSTM编码器
    print("Loading LSTM encoder...")
    lstm_encoder.load_model(lstm_model_path)
    
    # 加载ResNet编码器
    print("Loading ResNet encoder...")
    resnet_encoder.load_model(resnet_model_path)
    
    # 预处理音频
    audio_name = os.path.basename(audio_path).split(".")[0]
    print(f"Processing audio: {audio_path}")
    waveform = preprocess_audio(audio_path)
    
    # 计算嵌入向量
    print("Generating LSTM embedding...")
    lstm_embed = lstm_encoder.embed_utterance(waveform)
    
    print("Generating ResNet embedding...")
    resnet_embed = resnet_encoder.embed_utterance(waveform)
    
    # 可视化嵌入向量
    embeddings_path = output_dir / f"{audio_name}_embeddings_comparison.png"
    plot_embeddings_comparison(lstm_embed, resnet_embed, title=f"Audio: {audio_name}", save_path=embeddings_path)
    
    # 可视化梅尔频谱图
    mel_path = output_dir / f"{audio_name}_mel_spectrogram.png"
    plot_mel_spectrograms(audio_path, save_path=mel_path)
    
    return {
        "audio_name": audio_name,
        "lstm_embed": lstm_embed,
        "resnet_embed": resnet_embed,
        "embeddings_path": embeddings_path,
        "mel_path": mel_path
    }


def process_directory(audio_dir, lstm_model_path, resnet_model_path, output_dir, use_cpu=False, limit=5):
    """处理目录中的多个音频文件"""
    audio_dir = Path(audio_dir)
    
    # 支持的音频格式
    audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    
    # 查找音频文件
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(audio_dir.glob(f"*{ext}")))
    
    print(f"Found {len(audio_files)} audio files")
    
    # 限制处理数量
    if limit > 0 and len(audio_files) > limit:
        print(f"Limiting to the first {limit} files")
        audio_files = audio_files[:limit]
    
    # 处理每个音频文件
    results = []
    for audio_path in audio_files:
        try:
            result = process_audio_file(audio_path, lstm_model_path, resnet_model_path, output_dir, use_cpu)
            results.append(result)
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
    
    # 生成摘要报告
    similarities = [np.dot(r["lstm_embed"], r["resnet_embed"]) / 
                   (np.linalg.norm(r["lstm_embed"]) * np.linalg.norm(r["resnet_embed"])) 
                   for r in results]
    
    avg_similarity = np.mean(similarities)
    
    print("\n--- Summary Report ---")
    print(f"Processed {len(results)} audio files")
    print(f"Average cosine similarity between LSTM and ResNet encoders: {avg_similarity:.4f}")
    print(f"Max similarity: {max(similarities):.4f}, Min similarity: {min(similarities):.4f}")
    print(f"Visualization results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualization tool for comparing LSTM and ResNet encoder embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("audio_input", type=str, 
                        help="Path to an audio file or directory containing audio files")
    parser.add_argument("--lstm_model", type=Path, 
                        default="saved_models/default/encoder.pt",
                        help="Path to LSTM encoder model")
    parser.add_argument("--resnet_model", type=Path, 
                        default="saved_models/resnet_encoder/resnet_encoder.pt",
                        help="Path to ResNet encoder model")
    parser.add_argument("--output_dir", type=str, 
                        default="encoder_visualizations",
                        help="Path to output directory")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU for processing")
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum number of files to process from directory, 0 for unlimited")
    
    args = parser.parse_args()
    print_args(args, parser)
    
    # 检查输入是文件还是目录
    input_path = Path(args.audio_input)
    if input_path.is_file():
        # 处理单个文件
        process_audio_file(input_path, args.lstm_model, args.resnet_model, args.output_dir, args.cpu)
    elif input_path.is_dir():
        # 处理目录
        process_directory(input_path, args.lstm_model, args.resnet_model, args.output_dir, args.cpu, args.limit)
    else:
        print(f"Error: Input path {input_path} does not exist or is not a valid file/directory") 