#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import librosa

# 导入编码器模块
from encoder import inference as encoder
from ResNet.encoder_adapter import load_resnet_encoder, ResNetSpeakerEncoder

def parse_args():
    parser = argparse.ArgumentParser(
        description="评估ResNet编码器与原始LSTM编码器的性能比较",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--lstm_weights_path", type=Path, 
                        default="saved_models/default/encoder_lstm.pt",
                        help="原始LSTM编码器权重路径")
    parser.add_argument("--resnet_weights_path", type=Path, 
                        default="saved_models/default/encoder.pt",
                        help="ResNet编码器权重路径")
    parser.add_argument("--test_data_dir", type=Path, 
                        default="dataset/SV2TTS_test/encoder",
                        help="测试数据目录")
    parser.add_argument("--output_dir", type=Path, 
                        default="evaluation_results_train",
                        help="评估结果输出目录")
    parser.add_argument("--cpu", action="store_true", 
                        help="如果为真，将强制在CPU上运行")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批处理大小")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="评估样本数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()

def setup_device(args):
    """设置计算设备"""
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("使用CPU进行评估")
    else:
        device = torch.device("cuda")
        print(f"使用GPU进行评估: {torch.cuda.get_device_name(0)}")
    return device

def load_encoders(args, device):
    """加载LSTM和ResNet编码器"""
    print("加载编码器...")
    
    # 加载LSTM编码器
    lstm_encoder = encoder.SpeakerEncoder(device, device)
    lstm_checkpoint = torch.load(args.lstm_weights_path, device)
    
    # 处理权重文件中的额外参数
    if "model_state" in lstm_checkpoint:
        model_state = lstm_checkpoint["model_state"]
    else:
        model_state = lstm_checkpoint
    
    # 移除不匹配的参数
    current_state_dict = lstm_encoder.state_dict()
    filtered_state_dict = {k: v for k, v in model_state.items() if k in current_state_dict}
    
    # 加载匹配的参数
    lstm_encoder.load_state_dict(filtered_state_dict, strict=False)
    lstm_encoder.eval()
    print(f"已加载LSTM编码器: {args.lstm_weights_path}")
    
    # 加载ResNet编码器
    try:
        resnet_encoder = load_resnet_encoder(args.resnet_weights_path, device)
        print(f"已加载ResNet编码器: {args.resnet_weights_path}")
    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(e):
            print("检测到适配层模型，尝试使用兼容模式加载...")
            resnet_encoder = ResNetSpeakerEncoder(device, device)
            checkpoint = torch.load(args.resnet_weights_path, device)
            if "model_state" in checkpoint:
                resnet_encoder.load_state_dict(checkpoint["model_state"])
            else:
                resnet_encoder.load_state_dict(checkpoint)
        else:
            raise e
    
    resnet_encoder.eval()
    
    return lstm_encoder, resnet_encoder

def load_test_data(data_dir, num_samples=1000, batch_size=32):
    """加载测试数据"""
    print(f"加载测试数据: {data_dir}")
    
    # 获取所有音频文件
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".npy") and not file.startswith("_"):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"警告: 在 {data_dir} 中没有找到梅尔频谱图文件")
        print("尝试使用默认测试数据...")
        # 尝试使用默认测试数据
        default_data_dir = "dataset/SV2TTS/encoder"
        if os.path.exists(default_data_dir) and default_data_dir != data_dir:
            return load_test_data(default_data_dir, num_samples, batch_size)
        else:
            print("错误: 无法找到有效的测试数据")
            return [], []
    
    print(f"找到 {len(audio_files)} 个梅尔频谱图文件")
    
    # 随机选择样本
    np.random.shuffle(audio_files)
    audio_files = audio_files[:num_samples]
    
    # 按说话者分组
    speaker_files = {}
    for file in audio_files:
        speaker = os.path.basename(os.path.dirname(file))
        if speaker not in speaker_files:
            speaker_files[speaker] = []
        speaker_files[speaker].append(file)
    
    # 创建正负样本对
    positive_pairs = []
    negative_pairs = []
    
    # 正样本对：同一说话者的不同话语
    for speaker, files in speaker_files.items():
        if len(files) >= 2:
            for i in range(0, len(files), 2):
                if i + 1 < len(files):
                    positive_pairs.append((files[i], files[i+1]))
    
    # 负样本对：不同说话者的话语
    speakers = list(speaker_files.keys())
    for i, speaker1 in enumerate(speakers):
        for speaker2 in speakers[i+1:]:
            if speaker_files[speaker1] and speaker_files[speaker2]:
                negative_pairs.append((speaker_files[speaker1][0], speaker_files[speaker2][0]))
    
    # 平衡正负样本
    min_pairs = min(len(positive_pairs), len(negative_pairs))
    if min_pairs > 1000:  # 限制样本对数量，提高效率
        min_pairs = 1000
    positive_pairs = positive_pairs[:min_pairs]
    negative_pairs = negative_pairs[:min_pairs]
    
    print(f"创建了 {len(positive_pairs)} 对正样本和 {len(negative_pairs)} 对负样本")
    
    return positive_pairs, negative_pairs

def compute_embeddings(encoder, mel_files, device, batch_size=32):
    """计算嵌入向量"""
    embeddings = []
    
    for i in tqdm(range(0, len(mel_files), batch_size), desc="计算嵌入向量"):
        batch_files = mel_files[i:i+batch_size]
        batch_mels = []
        
        # 首先加载所有梅尔频谱图
        for file in batch_files:
            try:
                mel = np.load(file)
                batch_mels.append(mel)
            except Exception as e:
                print(f"加载文件错误 {file}: {e}")
                continue
        
        if not batch_mels:
            continue
        
        # 找出这个批次中最大的高度和宽度
        max_freq = max(mel.shape[0] for mel in batch_mels)
        max_time = max(mel.shape[1] for mel in batch_mels)
        
        # 填充到相同大小
        padded_mels = []
        for mel in batch_mels:
            # 检查是否需要在两个维度上填充
            if mel.shape[0] < max_freq or mel.shape[1] < max_time:
                pad_width = ((0, max_freq - mel.shape[0]), (0, max_time - mel.shape[1]))
                padded_mel = np.pad(mel, pad_width, mode='constant')
                padded_mels.append(padded_mel)
            else:
                padded_mels.append(mel)
                
        # 检查所有填充后的梅尔频谱图形状是否相同
        shapes = [mel.shape for mel in padded_mels]
        if len(set(shapes)) > 1:
            print(f"警告: 填充后的形状不一致: {shapes}")
            continue
        
        try:
            # 转换为张量
            batch_tensor = torch.FloatTensor(np.stack(padded_mels)).to(device)
            
            # 计算嵌入向量
            with torch.no_grad():
                batch_embeddings = encoder(batch_tensor)
            
            embeddings.extend(batch_embeddings.cpu().numpy())
        except Exception as e:
            print(f"处理批次错误: {e}")
            continue
    
    return embeddings

def compute_similarity_matrix(embeddings1, embeddings2):
    """计算相似度矩阵"""
    # 归一化嵌入向量
    embeddings1_norm = F.normalize(torch.FloatTensor(embeddings1), p=2, dim=1)
    embeddings2_norm = F.normalize(torch.FloatTensor(embeddings2), p=2, dim=1)
    
    # 计算余弦相似度
    similarity_matrix = torch.mm(embeddings1_norm, embeddings2_norm.t())
    
    return similarity_matrix.numpy()

def evaluate_encoder(lstm_encoder, resnet_encoder, positive_pairs, negative_pairs, device, batch_size=32):
    """评估编码器性能"""
    print("评估编码器性能...")
    
    # 检查是否有足够的样本
    if not positive_pairs or not negative_pairs:
        print("错误: 没有足够的样本进行评估")
        print("请确保测试数据目录中包含足够的梅尔频谱图文件")
        print("每个说话者至少需要2个样本，且至少需要2个不同的说话者")
        return None
    
    # 提取所有唯一的梅尔频谱图文件
    all_files = list(set([file for pair in positive_pairs + negative_pairs for file in pair]))
    
    # 计算嵌入向量
    print("计算LSTM编码器嵌入向量...")
    lstm_embeddings = compute_embeddings(lstm_encoder, all_files, device, batch_size)
    
    print("计算ResNet编码器嵌入向量...")
    resnet_embeddings = compute_embeddings(resnet_encoder, all_files, device, batch_size)
    
    # 创建文件到索引的映射
    file_to_idx = {file: idx for idx, file in enumerate(all_files)}
    
    # 计算相似度
    print("计算LSTM编码器相似度...")
    lstm_similarities = []
    lstm_labels = []
    
    for pair in positive_pairs:
        idx1, idx2 = file_to_idx[pair[0]], file_to_idx[pair[1]]
        similarity = F.cosine_similarity(
            torch.FloatTensor(lstm_embeddings[idx1]).unsqueeze(0),
            torch.FloatTensor(lstm_embeddings[idx2]).unsqueeze(0)
        ).item()
        lstm_similarities.append(similarity)
        lstm_labels.append(1)  # 正样本
    
    for pair in negative_pairs:
        idx1, idx2 = file_to_idx[pair[0]], file_to_idx[pair[1]]
        similarity = F.cosine_similarity(
            torch.FloatTensor(lstm_embeddings[idx1]).unsqueeze(0),
            torch.FloatTensor(lstm_embeddings[idx2]).unsqueeze(0)
        ).item()
        lstm_similarities.append(similarity)
        lstm_labels.append(0)  # 负样本
    
    print("计算ResNet编码器相似度...")
    resnet_similarities = []
    resnet_labels = []
    
    for pair in positive_pairs:
        idx1, idx2 = file_to_idx[pair[0]], file_to_idx[pair[1]]
        similarity = F.cosine_similarity(
            torch.FloatTensor(resnet_embeddings[idx1]).unsqueeze(0),
            torch.FloatTensor(resnet_embeddings[idx2]).unsqueeze(0)
        ).item()
        resnet_similarities.append(similarity)
        resnet_labels.append(1)  # 正样本
    
    for pair in negative_pairs:
        idx1, idx2 = file_to_idx[pair[0]], file_to_idx[pair[1]]
        similarity = F.cosine_similarity(
            torch.FloatTensor(resnet_embeddings[idx1]).unsqueeze(0),
            torch.FloatTensor(resnet_embeddings[idx2]).unsqueeze(0)
        ).item()
        resnet_similarities.append(similarity)
        resnet_labels.append(0)  # 负样本
    
    # 计算ROC曲线和AUC
    lstm_fpr, lstm_tpr, _ = roc_curve(lstm_labels, lstm_similarities)
    lstm_auc = auc(lstm_fpr, lstm_tpr)
    
    resnet_fpr, resnet_tpr, _ = roc_curve(resnet_labels, resnet_similarities)
    resnet_auc = auc(resnet_fpr, resnet_tpr)
    
    # 计算EER (Equal Error Rate)
    lstm_eer = compute_eer(lstm_similarities, lstm_labels)
    resnet_eer = compute_eer(resnet_similarities, resnet_labels)
    
    # 计算嵌入向量之间的相似度
    embedding_similarities = []
    for i in range(len(lstm_embeddings)):
        similarity = F.cosine_similarity(
            torch.FloatTensor(lstm_embeddings[i]).unsqueeze(0),
            torch.FloatTensor(resnet_embeddings[i]).unsqueeze(0)
        ).item()
        embedding_similarities.append(similarity)
    
    avg_embedding_similarity = np.mean(embedding_similarities)
    
    # 计算推理时间
    print("测量推理时间...")
    lstm_time = measure_inference_time(lstm_encoder, all_files[:batch_size], device)
    resnet_time = measure_inference_time(resnet_encoder, all_files[:batch_size], device)
    
    # 计算相似度统计信息
    lstm_similarity_stats = {
        "min": np.min(lstm_similarities),
        "max": np.max(lstm_similarities),
        "mean": np.mean(lstm_similarities),
        "std": np.std(lstm_similarities)
    }
    
    resnet_similarity_stats = {
        "min": np.min(resnet_similarities),
        "max": np.max(resnet_similarities),
        "mean": np.mean(resnet_similarities),
        "std": np.std(resnet_similarities)
    }
    
    return {
        "lstm": {
            "auc": lstm_auc,
            "eer": lstm_eer,
            "inference_time": lstm_time,
            "similarity_stats": lstm_similarity_stats
        },
        "resnet": {
            "auc": resnet_auc,
            "eer": resnet_eer,
            "inference_time": resnet_time,
            "similarity_stats": resnet_similarity_stats
        },
        "embedding_similarity": avg_embedding_similarity,
        "embedding_similarities": embedding_similarities,
        "lstm_fpr": lstm_fpr,
        "lstm_tpr": lstm_tpr,
        "resnet_fpr": resnet_fpr,
        "resnet_tpr": resnet_tpr,
        "lstm_similarities": np.array(lstm_similarities),
        "lstm_labels": np.array(lstm_labels),
        "resnet_similarities": np.array(resnet_similarities),
        "resnet_labels": np.array(resnet_labels)
    }

def compute_eer(similarities, labels):
    """计算等错误率 (EER)"""
    # 将相似度和标签转换为numpy数组
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # 获取正负样本的相似度
    pos_similarities = similarities[labels == 1]
    neg_similarities = similarities[labels == 0]
    
    # 计算所有可能的阈值
    thresholds = np.sort(np.concatenate([pos_similarities, neg_similarities]))
    
    # 计算每个阈值的FAR和FRR
    min_eer = 1.0
    eer_threshold = 0.0
    
    for threshold in thresholds:
        # 假接受率 (FAR) = 负样本被错误接受的比率
        far = np.mean(neg_similarities >= threshold)
        
        # 假拒绝率 (FRR) = 正样本被错误拒绝的比率
        frr = np.mean(pos_similarities < threshold)
        
        # 计算EER
        eer = (far + frr) / 2
        
        if eer < min_eer:
            min_eer = eer
            eer_threshold = threshold
    
    return min_eer

def measure_inference_time(encoder, mel_files, device, num_runs=10):
    """测量推理时间"""
    # 准备一个批次的数据
    batch_mels = []
    for file in mel_files:
        try:
            mel = np.load(file)
            batch_mels.append(mel)
        except Exception as e:
            print(f"加载文件错误 {file}: {e}")
            continue
    
    if not batch_mels:
        print("警告: 没有有效的梅尔频谱图用于测量推理时间")
        return 0.0
    
    # 找出这个批次中最大的高度和宽度
    max_freq = max(mel.shape[0] for mel in batch_mels)
    max_time = max(mel.shape[1] for mel in batch_mels)
    
    # 填充到相同大小
    padded_mels = []
    for mel in batch_mels:
        # 检查是否需要在两个维度上填充
        if mel.shape[0] < max_freq or mel.shape[1] < max_time:
            pad_width = ((0, max_freq - mel.shape[0]), (0, max_time - mel.shape[1]))
            padded_mel = np.pad(mel, pad_width, mode='constant')
            padded_mels.append(padded_mel)
        else:
            padded_mels.append(mel)
    
    # 检查所有填充后的梅尔频谱图形状是否相同
    shapes = [mel.shape for mel in padded_mels]
    if len(set(shapes)) > 1:
        print(f"警告: 填充后的形状不一致: {shapes}")
        return 0.0
    
    try:
        # 转换为张量
        batch_tensor = torch.FloatTensor(np.stack(padded_mels)).to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = encoder(batch_tensor)
        
        # 测量推理时间
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = encoder(batch_tensor)
                end_time = time.time()
                times.append(end_time - start_time)
        
        # 计算平均推理时间
        avg_time = np.mean(times)
        
        return avg_time
    except Exception as e:
        print(f"测量推理时间错误: {e}")
        return 0.0

def plot_results(results, output_dir):
    """绘制评估结果图表"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置matplotlib的字体和样式
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(results["lstm_fpr"], results["lstm_tpr"], 
             label=f"LSTM (AUC = {results['lstm']['auc']:.4f})")
    plt.plot(results["resnet_fpr"], results["resnet_tpr"], 
             label=f"ResNet (AUC = {results['resnet']['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()
    
    # 绘制嵌入向量相似度分布
    plt.figure(figsize=(10, 8))
    plt.hist(results["lstm_similarities"][results["lstm_labels"] == 1], 
             bins=50, alpha=0.5, label="LSTM Positive", density=True)
    plt.hist(results["lstm_similarities"][results["lstm_labels"] == 0], 
             bins=50, alpha=0.5, label="LSTM Negative", density=True)
    plt.hist(results["resnet_similarities"][results["resnet_labels"] == 1], 
             bins=50, alpha=0.5, label="ResNet Positive", density=True)
    plt.hist(results["resnet_similarities"][results["resnet_labels"] == 0], 
             bins=50, alpha=0.5, label="ResNet Negative", density=True)
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.title("Embedding Similarity Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "embedding_similarity.png"))
    plt.close()
    
    # 保存评估结果到文本文件
    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write("Encoder Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("LSTM Encoder:\n")
        f.write("-" * 20 + "\n")
        f.write(f"AUC: {results['lstm']['auc']:.4f}\n")
        f.write(f"EER: {results['lstm']['eer']:.4f}\n")
        f.write(f"Average Inference Time: {results['lstm']['inference_time']:.4f} seconds\n")
        f.write("\nSimilarity Statistics:\n")
        f.write(f"Min: {results['lstm']['similarity_stats']['min']:.4f}\n")
        f.write(f"Max: {results['lstm']['similarity_stats']['max']:.4f}\n")
        f.write(f"Mean: {results['lstm']['similarity_stats']['mean']:.4f}\n")
        f.write(f"Std: {results['lstm']['similarity_stats']['std']:.4f}\n")
        
        f.write("\nResNet Encoder:\n")
        f.write("-" * 20 + "\n")
        f.write(f"AUC: {results['resnet']['auc']:.4f}\n")
        f.write(f"EER: {results['resnet']['eer']:.4f}\n")
        f.write(f"Average Inference Time: {results['resnet']['inference_time']:.4f} seconds\n")
        f.write("\nSimilarity Statistics:\n")
        f.write(f"Min: {results['resnet']['similarity_stats']['min']:.4f}\n")
        f.write(f"Max: {results['resnet']['similarity_stats']['max']:.4f}\n")
        f.write(f"Mean: {results['resnet']['similarity_stats']['mean']:.4f}\n")
        f.write(f"Std: {results['resnet']['similarity_stats']['std']:.4f}\n")
        
        f.write("\nPerformance Comparison:\n")
        f.write("-" * 20 + "\n")
        f.write(f"AUC Improvement: {(results['resnet']['auc'] - results['lstm']['auc']):.4f}\n")
        f.write(f"EER Improvement: {(results['lstm']['eer'] - results['resnet']['eer']):.4f}\n")
        f.write(f"Speed Improvement: {(results['lstm']['inference_time'] / results['resnet']['inference_time']):.2f}x\n")

def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置设备
    device = setup_device(args)
    
    # 加载编码器
    lstm_encoder, resnet_encoder = load_encoders(args, device)
    
    # 加载测试数据
    positive_pairs, negative_pairs = load_test_data(
        args.test_data_dir, 
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    # 评估编码器
    results = evaluate_encoder(
        lstm_encoder, 
        resnet_encoder, 
        positive_pairs, 
        negative_pairs, 
        device,
        batch_size=args.batch_size
    )
    
    # 检查结果是否有效
    if results is None:
        print("评估失败，请检查测试数据")
        return
    
    # 绘制结果
    plot_results(results, args.output_dir)
    
    print(f"评估完成! 结果已保存到 {args.output_dir}")

if __name__ == "__main__":
    main() 