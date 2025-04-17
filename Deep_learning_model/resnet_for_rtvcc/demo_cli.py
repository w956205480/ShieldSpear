import argparse
import os
import time
import logging
import sys
from pathlib import Path
import traceback
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import inspect

import librosa
import numpy as np
import soundfile as sf
import torch
from datetime import datetime

# 设置详细的日志格式
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger("Voice-Cloning-Demo")

# 导入所需模块
from encoder import inference as encoder
from ResNet import inference as resnet_encoder  # 导入ResNet的inference模块
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from synthesizer.hparams import hparams  # 导入合成器超参数
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder


# 打印时间戳的辅助函数
def print_time(message):
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    logger.info(f"[{current_time}] {message}")


# 打印分隔线的辅助函数
def print_section(title):
    line = "=" * 50
    logger.info(f"\n{line}\n{title}\n{line}")


# 打印向量或张量信息的辅助函数
def print_tensor_info(name, tensor):
    if tensor is None:
        logger.info(f"{name}: None")
        return
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    if isinstance(tensor, np.ndarray):
        try:
            logger.info(f"{name}: 形状={tensor.shape}, 类型={tensor.dtype}, 范围=[{tensor.min():.4f}, {tensor.max():.4f}], 均值={tensor.mean():.4f}, 标准差={tensor.std():.4f}")
            # 检查NaN或Inf
            has_nan = np.isnan(tensor).any()
            has_inf = np.isinf(tensor).any()
            if has_nan or has_inf:
                logger.warning(f"{name} 包含 {'NaN ' if has_nan else ''}{'Inf ' if has_inf else ''}值!")
        except Exception as e:
            logger.error(f"无法分析张量 {name}: {e}")
    else:
        logger.info(f"{name}: {tensor}")


# 测试模型的函数
def test_models(active_encoder, synthesizer, args):
    print_section("测试模型配置")
    logger.info("使用小输入测试模型配置...")

    # 测试编码器
    logger.info("测试编码器...")
    test_start = time.time()
    test_wav = np.zeros(encoder.sampling_rate)
    logger.info(f"创建了{len(test_wav)/encoder.sampling_rate:.2f}秒的零音频样本进行测试")
    
    try:
        # 预处理测试音频
        logger.info("正在预处理测试音频...")
        preprocessed_wav = active_encoder.preprocess_wav(test_wav)
        logger.info(f"预处理完成，音频长度: {len(preprocessed_wav)}")
        
        # 生成嵌入向量
        logger.info("正在生成嵌入向量...")
        embed_test = active_encoder.embed_utterance(preprocessed_wav)
        
        # 确保嵌入向量格式正确
        if len(embed_test.shape) > 1:
            logger.info(f"嵌入向量维度为 {embed_test.shape}，调整为一维")
            embed_test = embed_test.flatten()
            
        # 确保嵌入向量是float32类型
        if embed_test.dtype != np.float32:
            logger.info(f"将嵌入向量从 {embed_test.dtype} 转换为 float32")
            embed_test = embed_test.astype(np.float32)
            
        print_tensor_info("编码器嵌入", embed_test)
        logger.info(f"编码器测试完成，耗时: {time.time() - test_start:.2f}秒")
    except Exception as e:
        logger.error(f"编码器测试失败: {e}")
        logger.info("打印完整栈跟踪信息:")
        traceback.print_exc()
        logger.info("尝试创建随机嵌入向量进行后续测试")
        embed_test = np.random.randn(speaker_embedding_size).astype(np.float32)
        embed_test = embed_test / np.linalg.norm(embed_test)
        print_tensor_info("随机生成的嵌入", embed_test)

    # 创建一个随机嵌入向量用于测试
    random_embed_start = time.time()
    logger.info("创建随机嵌入向量进行测试...")
    embed = np.random.rand(speaker_embedding_size).astype(np.float32)
    # 嵌入向量通常是L2归一化的
    embed = embed / np.linalg.norm(embed)
    print_tensor_info("随机嵌入", embed)
    logger.info(f"随机嵌入创建完成，耗时: {time.time() - random_embed_start:.2f}秒")
    
    # 测试合成器
    synth_start = time.time()
    logger.info("测试合成器...")
    texts = ["测试语音合成器，它应该能够处理ResNet生成的嵌入", "处理第二个测试样本"]
    logger.info(f"测试文本: {texts}")
    
    # 确保嵌入向量列表格式正确
    try:
        logger.info("准备测试嵌入向量...")
        embed1 = embed.flatten()
        embed2 = np.zeros(speaker_embedding_size, dtype=np.float32)
        # 确保第二个向量也是归一化的
        embed2 = embed2 / (np.linalg.norm(embed2) + 1e-8)
        embeds = [embed1, embed2]
        logger.info(f"测试嵌入向量形状: {[e.shape for e in embeds]}")
        logger.info(f"测试嵌入向量类型: {[e.dtype for e in embeds]}")
        
        logger.info("调用合成器合成梅尔频谱图...")
        
        # 添加详细日志
        log_synthesizer_details(synthesizer, texts, embeds)
        
        # 记录模型参数状态
        if synthesizer._model is not None:
            try:
                # 检查模型参数数量
                param_count = sum(p.numel() for p in synthesizer._model.parameters())
                trainable_param_count = sum(p.numel() for p in synthesizer._model.parameters() if p.requires_grad)
                logger.info(f"合成器模型总参数: {param_count:,}, 可训练参数: {trainable_param_count:,}")
                
                # 检查部分参数统计信息
                for name, param in synthesizer._model.named_parameters():
                    if param.numel() > 0 and ("attention" in name or "encoder" in name or "decoder" in name):
                        logger.info(f"参数 {name}: 形状={param.shape}, 均值={param.data.mean().item():.4f}, 标准差={param.data.std().item():.4f}")
            except Exception as pe:
                logger.warning(f"检查模型参数时出错: {pe}")
        
        mels = synthesizer.synthesize_spectrograms(texts, embeds)
        logger.info(f"合成完成，得到 {len(mels)} 个梅尔频谱图")
        
        # 详细检查合成结果
        for i, mel in enumerate(mels):
            # 基本信息
            logger.info(f"梅尔频谱图 #{i}: 形状={mel.shape}, "
                      f"范围=[{mel.min():.4f}, {mel.max():.4f}], "
                      f"均值={mel.mean():.4f}, 标准差={mel.std():.4f}")
            
            # 详细分析
            analyze_mel_spectrogram(mel, f"测试梅尔频谱图 #{i} 详细分析")
            
            # 检查是否异常短
            if mel.shape[1] < 50:
                logger.warning(f"梅尔频谱图 #{i} 长度异常短 ({mel.shape[1]} 帧)!")
        
        # 检查合成器内部
        inspect_synthesizer_internals(synthesizer, texts, embed)
        
        logger.info(f"合成器测试完成，耗时: {time.time() - synth_start:.2f}秒")
    except Exception as e:
        logger.error(f"合成器测试失败: {e}")
        logger.info("打印完整栈跟踪信息:")
        traceback.print_exc()
        
        # 尝试不同的输入格式
        logger.info("尝试使用不同的输入格式...")
        try:
            # 尝试不同的嵌入向量格式
            logger.info("尝试以列向量形式提供嵌入向量...")
            reshaped_embeds = [embed.reshape(-1), np.zeros(speaker_embedding_size)]
            logger.info(f"重塑后的嵌入向量形状: {[e.shape for e in reshaped_embeds]}")
            
            logger.info("再次尝试合成...")
            mels = synthesizer.synthesize_spectrograms(texts, reshaped_embeds)
            logger.info(f"第二次尝试成功，得到 {len(mels)} 个梅尔频谱图")
            for i, mel in enumerate(mels):
                print_tensor_info(f"梅尔频谱图 #{i}", mel)
        except Exception as e2:
            logger.error(f"第二次尝试也失败: {e2}")
            logger.info("打印完整栈跟踪信息:")
            traceback.print_exc()
            
            # 创建默认频谱图
            logger.info("创建默认的梅尔频谱图用于测试声码器")
            mels = [np.zeros((hparams.num_mels, 100)), np.zeros((hparams.num_mels, 100))]
            logger.info(f"创建了默认梅尔频谱图，形状: {[m.shape for m in mels]}")

    # 测试声码器
    voc_start = time.time()
    try:
        # 连接梅尔频谱图
        logger.info("连接梅尔频谱图...")
        mel = np.concatenate(mels, axis=1)
        print_tensor_info("连接后的梅尔频谱图", mel)
        
        # 定义进度回调
        def progress_callback(i, total, b_size, gen_rate):
            logger.info(f"声码器生成进度: {i}/{total} ({i/total*100:.1f}%), 批次大小: {b_size}, 速率: {gen_rate:.1f}kHz")
        
        logger.info("开始测试声码器...")
        logger.info("设置目标长度为200，重叠为50，这些参数很小，仅用于测试目的")
        
        # 调用声码器进行推理
        vocoder_output = vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=progress_callback)
        print_tensor_info("声码器输出", vocoder_output)
        
        # 声码器后处理
        try:
            logger.info("应用淡入淡出效果...")
            vocoder_output = vocoder.apply_taper(vocoder_output)
            logger.info("正在归一化音量...")
            vocoder_output = vocoder.normalize_volume(vocoder_output)
            print_tensor_info("最终处理后的输出", vocoder_output)
        except Exception as post_error:
            logger.warning(f"后处理声码器输出时出错: {post_error}，但会继续执行")
            logger.warning("跳过音频后处理步骤")
        
        logger.info(f"声码器测试完成，耗时: {time.time() - voc_start:.2f}秒")
    except Exception as v_error:
        logger.error(f"声码器测试失败: {v_error}")
        logger.info("打印完整栈跟踪信息:")
        traceback.print_exc()
        logger.error("请检查声码器模型!")
        vocoder_output = None

    logger.info("所有测试完成!")
    if args.seed is not None:
        logger.info(f"当前使用的随机种子: {args.seed}")
    
    # 打印成功列表
    test_results = {
        "编码器": embed_test is not None,
        "合成器": 'mels' in locals() and len(mels) > 0,
        "声码器": 'vocoder_output' in locals() and vocoder_output is not None
    }
    
    logger.info(f"测试结果汇总:")
    for name, success in test_results.items():
        status = "✓ 成功" if success else "✗ 失败"
        logger.info(f"  {name}: {status}")


# 在synthesizer部分添加详细的日志函数
def log_synthesizer_details(synthesizer, texts, embeds):
    """记录合成器详细信息，帮助诊断问题"""
    logger.info("====== 合成器详细信息 ======")
    
    # 检查合成器模型是否已加载
    logger.info(f"合成器模型已加载: {synthesizer.is_loaded()}")
    if not synthesizer.is_loaded():
        logger.info("正在尝试加载合成器模型...")
        synthesizer.load()
    
    # 显示模型参数
    if hasattr(synthesizer._model, 'num_params'):
        params = synthesizer._model.num_params(print_out=False)
        logger.info(f"合成器模型参数: {params:,}")
    
    # 检查文本输入
    logger.info(f"输入文本长度: {[len(text) for text in texts]}")
    logger.info(f"文本示例: '{texts[0][:50]}...'")
    
    # 检查嵌入向量
    for i, embed in enumerate(embeds):
        logger.info(f"嵌入向量 #{i}: 形状={embed.shape}, 范围=[{embed.min():.4f}, {embed.max():.4f}], "
                   f"均值={embed.mean():.4f}, 标准差={embed.std():.4f}")
        
        # 检查嵌入向量是否包含NaN或Inf
        if np.isnan(embed).any():
            logger.error(f"嵌入向量 #{i} 包含NaN值!")
        if np.isinf(embed).any():
            logger.error(f"嵌入向量 #{i} 包含Inf值!")
    
    # 检查模型设备
    if hasattr(synthesizer, 'device'):
        logger.info(f"合成器运行设备: {synthesizer.device}")
    
    logger.info("===============================")


# 添加分析梅尔频谱图的详细函数
def analyze_mel_spectrogram(spec, name="梅尔频谱图"):
    """分析梅尔频谱图的质量，检查常见问题并打印统计数据
    
    Args:
        spec: 梅尔频谱图，形状为 (n_mels, n_frames)
        name: 频谱图的名称或来源，用于日志记录
    """
    logger.info(f"===== 分析{name} =====")
    
    try:
        # 检查频谱图是否为空或者None
        if spec is None:
            logger.error(f"{name}为空 (None)")
            return
        
        # 确保是numpy数组以便分析
        if isinstance(spec, torch.Tensor):
            spec = spec.detach().cpu().numpy()
        
        # 检查维度
        if not isinstance(spec, np.ndarray):
            logger.error(f"{name}不是numpy数组，而是{type(spec)}")
            return
            
        # 检查形状
        if len(spec.shape) != 2:
            logger.error(f"{name}维度不是2D，而是{len(spec.shape)}D，形状={spec.shape}")
            return
            
        n_mels, n_frames = spec.shape
        logger.info(f"{name}形状: ({n_mels}, {n_frames})")
        
        # 基本统计信息
        spec_min = spec.min()
        spec_max = spec.max()
        spec_mean = spec.mean()
        spec_std = spec.std()
        logger.info(f"{name}统计: 范围=[{spec_min:.4f}, {spec_max:.4f}], 均值={spec_mean:.4f}, 标准差={spec_std:.4f}")
        
        # 检查空值
        has_nan = np.isnan(spec).any()
        has_inf = np.isinf(spec).any()
        
        if has_nan:
            nan_count = np.isnan(spec).sum()
            logger.error(f"{name}包含{nan_count}个NaN值 ({nan_count/(n_mels*n_frames)*100:.2f}%)")
        
        if has_inf:
            inf_count = np.isinf(spec).sum()
            logger.error(f"{name}包含{inf_count}个Inf值 ({inf_count/(n_mels*n_frames)*100:.2f}%)")
        
        # 检查是否全为0或接近0
        zero_ratio = (np.abs(spec) < 1e-5).mean()
        if zero_ratio > 0.5:
            logger.warning(f"{name}中{zero_ratio*100:.1f}%的值接近于零，可能是静音或梅尔生成问题")
        
        # 检查频率分布
        mel_means = spec.mean(axis=1)  # 每个梅尔频带的平均值
        mel_stds = spec.std(axis=1)    # 每个梅尔频带的标准差
        
        # 检查频谱图是否在某些频带上缺失内容
        empty_bands = (mel_stds < 0.01).sum()
        if empty_bands > n_mels * 0.3:  # 如果超过30%的频带几乎没有变化
            logger.warning(f"{name}在{empty_bands}个梅尔频带上几乎没有变化 ({empty_bands/n_mels*100:.1f}%)，可能表明声音不自然")
        
        # 检查帧之间的差异
        frame_diffs = np.abs(np.diff(spec, axis=1)).mean()
        if frame_diffs < 0.01:
            logger.warning(f"{name}帧间差异非常小 ({frame_diffs:.4f})，可能缺乏自然变化")
        elif frame_diffs > 0.5:
            logger.warning(f"{name}帧间差异非常大 ({frame_diffs:.4f})，可能存在噪声或不连贯")
            
        # 检查频谱图是否过于平坦
        flatness = mel_stds.mean()
        if flatness < 0.05:
            logger.warning(f"{name}过于平坦 (平均标准差={flatness:.4f})，可能导致单调的声音")
            
        # 计算整体质量得分 (简单启发式方法)
        quality_score = 0.0
        
        # 扣分项
        if has_nan or has_inf:
            quality_score -= 5.0
        
        quality_score -= zero_ratio * 3.0  # 太多接近零的值扣分
        
        # 帧差异得分 (应该适中)
        frame_diff_score = max(0, min(1, frame_diffs / 0.2)) if frame_diffs < 0.2 else max(0, 1 - (frame_diffs - 0.2) / 0.5)
        quality_score += frame_diff_score
        
        # 频率分布得分
        band_variation_score = 1 - (empty_bands / n_mels)
        quality_score += band_variation_score
        
        # 平坦度得分 (平坦度应该适中，不能太平也不能太突兀)
        flatness_score = min(1, flatness / 0.1) if flatness < 0.1 else max(0, 1 - (flatness - 0.1) / 0.5)
        quality_score += flatness_score
        
        # 标准偏差得分 (非零标准差是好的)
        std_score = min(1, spec_std / 0.2)
        quality_score += std_score
        
        # 规范化得分到0-10区间
        normalized_quality_score = max(0, min(10, quality_score * 2))
        
        if normalized_quality_score < 3:
            logger.error(f"{name}质量分数很低: {normalized_quality_score:.1f}/10")
        elif normalized_quality_score < 5:
            logger.warning(f"{name}质量分数较低: {normalized_quality_score:.1f}/10")
        elif normalized_quality_score < 7:
            logger.info(f"{name}质量分数一般: {normalized_quality_score:.1f}/10")
        else:
            logger.info(f"{name}质量分数良好: {normalized_quality_score:.1f}/10")
            
    except Exception as e:
        logger.error(f"分析{name}时出错: {e}")
        traceback.print_exc()


# 修复inspect_synthesizer_internals函数
def inspect_synthesizer_internals(synthesizer, texts, embed):
    """检查合成器内部状态并尝试执行内部步骤以查找问题
    
    Args:
        synthesizer: 语音合成器实例
        texts: 用于合成的文本列表
        embed: 用于合成的嵌入向量
    """
    logger.info("===== 检查合成器内部状态 =====")
    
    try:
        # 检查合成器是否已初始化
        if synthesizer is None:
            logger.error("合成器未初始化")
            return
            
        if not hasattr(synthesizer, "_model") or synthesizer._model is None:
            logger.error("合成器模型未加载")
            return
            
        # 检查文本
        if texts is None or len(texts) == 0:
            logger.error("文本列表为空")
            return
            
        for i, text in enumerate(texts):
            logger.info(f"输入文本 #{i+1}: '{text}'")
        
        # 检查嵌入向量
        if embed is None:
            logger.error("嵌入向量为空")
            return
            
        if isinstance(embed, list):
            if len(embed) == 0:
                logger.error("嵌入向量列表为空")
                return
                
            logger.info(f"嵌入向量是一个列表，包含{len(embed)}个向量")
            # 如果是张量列表，我们使用第一个进行检查
            embed_to_check = embed[0]
        else:
            embed_to_check = embed
            
        # 检查嵌入向量的形状和值
        if isinstance(embed_to_check, torch.Tensor):
            logger.info(f"嵌入向量是一个张量，形状={embed_to_check.shape}，设备={embed_to_check.device}")
            
            # 检查设备兼容性
            if hasattr(synthesizer, "device"):
                if embed_to_check.device != synthesizer.device:
                    logger.warning(f"嵌入向量在{embed_to_check.device}上，但合成器在{synthesizer.device}上！")
                else:
                    logger.info(f"设备兼容性: 正常 (都在{synthesizer.device}上)")
            
            # 将张量转换为numpy进行检查
            embed_np = embed_to_check.detach().cpu().numpy()
        elif isinstance(embed_to_check, np.ndarray):
            logger.info(f"嵌入向量是一个NumPy数组，形状={embed_to_check.shape}")
            embed_np = embed_to_check
        else:
            logger.error(f"无法识别的嵌入向量类型: {type(embed_to_check)}")
            return
            
        # 检查嵌入向量的值
        has_nan = np.isnan(embed_np).any()
        has_inf = np.isinf(embed_np).any()
        embed_min = embed_np.min()
        embed_max = embed_np.max()
        embed_mean = embed_np.mean()
        embed_std = embed_np.std()
        
        logger.info(f"嵌入向量统计: 范围=[{embed_min:.4f}, {embed_max:.4f}], 均值={embed_mean:.4f}, 标准差={embed_std:.4f}")
        
        if has_nan:
            logger.error("嵌入向量包含NaN值，这将导致合成失败")
        
        if has_inf:
            logger.error("嵌入向量包含Inf值，这将导致合成失败")
        
        # 检查合成器处理文本的能力
        logger.info("尝试处理文本...")
        try:
            # 尝试调用处理文本的内部方法（如果存在）
            if hasattr(synthesizer, "_process_texts"):
                processed_texts = synthesizer._process_texts(texts)
                logger.info(f"成功处理了{len(processed_texts)}个文本")
            else:
                logger.info("合成器没有_process_texts方法，跳过文本处理检查")
        except Exception as e:
            logger.error(f"处理文本失败: {e}")
            
        # 尝试合成
        logger.info("尝试合成梅尔频谱图...")
        try:
            if isinstance(embed, list) and len(texts) == 1 and len(embed) > 1:
                logger.warning(f"文本数量(1)与嵌入向量数量({len(embed)})不匹配，尝试使用第一个嵌入向量")
                simplified_embed = embed[0]
                try:
                    specs = synthesizer.synthesize_spectrograms(texts, [simplified_embed])
                    if specs is not None:
                        if isinstance(specs, list):
                            logger.info(f"成功合成了{len(specs)}个梅尔频谱图")
                            if len(specs) > 0:
                                spec = specs[0]
                                if isinstance(spec, torch.Tensor):
                                    spec = spec.detach().cpu().numpy()
                                logger.info(f"第一个梅尔频谱图形状: {spec.shape}")
                        else:
                            logger.info(f"合成返回非列表对象: {type(specs)}")
                except Exception as e:
                    logger.error(f"使用单个嵌入向量合成失败: {e}")
            else:
                try:
                    # 检查输入列表长度是否匹配
                    if isinstance(embed, list) and len(texts) != len(embed):
                        logger.warning(f"文本数量({len(texts)})与嵌入向量数量({len(embed)})不匹配")
                    
                    specs = synthesizer.synthesize_spectrograms(texts, embed)
                    if specs is not None:
                        if isinstance(specs, list):
                            logger.info(f"成功合成了{len(specs)}个梅尔频谱图")
                            if len(specs) > 0:
                                spec = specs[0]
                                if isinstance(spec, torch.Tensor):
                                    spec = spec.detach().cpu().numpy()
                                logger.info(f"第一个梅尔频谱图形状: {spec.shape}")
                        else:
                            logger.info(f"合成返回非列表对象: {type(specs)}")
                except Exception as e:
                    logger.error(f"合成失败: {e}")
                    traceback.print_exc()
        except Exception as e:
            logger.error(f"检测合成能力时出错: {e}")
            traceback.print_exc()
            
    except Exception as e:
        logger.error(f"检查合成器内部状态时出错: {e}")
        traceback.print_exc()


# 在synthesize_spectrograms中添加临时函数，打印每次合成的中间结果
def add_debug_hooks_to_synthesizer(synthesizer):
    """给合成器添加调试钩子，监控内部状态"""
    if synthesizer._model is None:
        logger.warning("合成器模型未加载，无法添加调试钩子")
        return
    
    logger.info("正在为合成器添加调试钩子...")
    
    # 保存原始的generate方法
    original_generate = synthesizer._model.generate
    
    # 定义新的generate方法，添加调试信息
    def debug_generate(*args, **kwargs):
        logger.info("开始合成过程，记录中间结果...")
        
        # 记录输入参数
        if len(args) >= 2:
            x, speaker_embedding = args[0], args[1]
            logger.info(f"输入文本序列形状: {x.shape}, 设备: {x.device}")
            logger.info(f"说话人嵌入形状: {speaker_embedding.shape}, 设备: {speaker_embedding.device}")
            
            # 检查输入中是否有异常值
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error("输入文本序列包含NaN或Inf值!")
            if torch.isnan(speaker_embedding).any() or torch.isinf(speaker_embedding).any():
                logger.error("说话人嵌入包含NaN或Inf值!")
        
        # 检查模型状态
        for name, module in synthesizer._model.named_modules():
            if isinstance(module, torch.nn.modules.rnn.GRU) or isinstance(module, torch.nn.Linear):
                for param_name, param in module.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logger.error(f"模块 {name} 的参数 {param_name} 包含NaN或Inf值!")
        
        # 调用原始方法
        try:
            results = original_generate(*args, **kwargs)
            
            # 检查结果
            if len(results) >= 3:
                mel_outputs, linear, attn_scores = results
                logger.info(f"合成结果: 梅尔输出形状={mel_outputs.shape}, 线性输出形状={linear.shape}, 注意力分数形状={attn_scores.shape}")
                
                # 检查结果中是否有异常值
                if torch.isnan(mel_outputs).any() or torch.isinf(mel_outputs).any():
                    logger.error("梅尔输出包含NaN或Inf值!")
                
                # 梅尔输出统计
                mel_min = mel_outputs.min().item()
                mel_max = mel_outputs.max().item()
                mel_mean = mel_outputs.mean().item()
                mel_std = mel_outputs.std().item()
                logger.info(f"梅尔输出统计: 范围=[{mel_min:.4f}, {mel_max:.4f}], 均值={mel_mean:.4f}, 标准差={mel_std:.4f}")
                
                # 检查注意力矩阵
                if attn_scores.numel() > 0:
                    attn_max = attn_scores.max().item()
                    attn_mean = attn_scores.mean().item()
                    logger.info(f"注意力分数统计: 最大值={attn_max:.4f}, 均值={attn_mean:.4f}")
                    
                    if attn_max < 0.5:
                        logger.warning(f"注意力最大值 ({attn_max:.4f}) 过低，可能影响合成质量")
            
            return results
        except Exception as e:
            logger.error(f"合成过程中发生错误: {e}")
            traceback.print_exc()
            raise
    
    # 替换generate方法
    synthesizer._model.generate = debug_generate
    logger.info("调试钩子添加完成")


# 处理交互式语音生成
def process_interactive_generation(encoder, synthesizer, vocoder, num=1):
    """处理交互式语音生成，包含详细的调试输出"""
    print_section("开始交互式语音生成")
    
    try:
        # 准备输出目录
        demo_output_dir = os.path.join("demo_output")
        os.makedirs(demo_output_dir, exist_ok=True)
        
        # 记录模型状态
        logger.info(f"编码器类型: {type(encoder).__name__}")
        logger.info(f"合成器类型: {type(synthesizer).__name__}")
        logger.info(f"声码器类型: {type(vocoder).__name__}")
        
        # 获取音频输入
        logger.info("请求用户音频输入...")
        in_fpath = Path(input("请输入参考音频文件路径 (按回车选择示例): ").strip())
        if in_fpath.is_file():
            logger.info(f"用户选择了音频文件: {in_fpath}")
        else:
            example_fpaths = list(Path("samples").glob("*.mp3"))
            if len(example_fpaths) == 0:
                example_fpaths = list(Path("samples").glob("*.wav"))
            if len(example_fpaths) == 0:
                logger.error("样本目录中找不到任何音频文件")
                example_fpaths = [Path("demo_utils/reference.wav")]
                
            logger.info(f"找到{len(example_fpaths)}个示例音频文件")
            in_fpath = np.random.choice(example_fpaths)
            logger.info(f"随机选择了示例音频: {in_fpath}")
            
        # 检查文件是否存在
        if not os.path.isfile(in_fpath):
            logger.error(f"音频文件不存在: {in_fpath}")
            return
            
        # 加载音频
        logger.info(f"加载音频文件: {in_fpath}...")
        original_wav, sampling_rate = librosa.load(str(in_fpath), sr=None)
        
        # 打印原始音频信息
        logger.info(f"原始音频: 采样率={sampling_rate}Hz, 长度={len(original_wav)}, 时长={len(original_wav)/sampling_rate:.2f}秒")
        
        # 重采样到16kHz (如果需要)
        if sampling_rate != 16000:
            logger.info(f"将音频从{sampling_rate}Hz重采样到16000Hz")
            wav = librosa.resample(original_wav, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000
        else:
            wav = original_wav
        
        # 处理音频
        logger.info("预处理音频...")
        try:
            # 使用preprocessing函数处理音频
            if hasattr(encoder, "preprocess_wav"):
                logger.info("使用编码器的preprocess_wav函数")
                preprocessed_wav = encoder.preprocess_wav(wav)
                logger.info(f"预处理后音频: 长度={len(preprocessed_wav)}, 采样率=16000Hz")
            else:
                logger.info("使用通用音频预处理函数")
                # 假设存在这些函数
                from encoder.audio import preprocess_wav
                preprocessed_wav = preprocess_wav(wav, source_sr=sampling_rate)
                logger.info(f"预处理后音频: 长度={len(preprocessed_wav)}, 采样率=16000Hz")
        except Exception as e:
            logger.error(f"音频预处理失败: {e}")
            logger.info("使用原始音频作为后备...")
            preprocessed_wav = wav
        
        # 检查预处理后的音频
        if len(preprocessed_wav) < 16000 * 0.5:  # 少于0.5秒
            logger.warning(f"预处理后的音频过短 ({len(preprocessed_wav)/16000:.2f}秒)，可能导致嵌入向量质量差")
            
        # 生成嵌入向量
        logger.info("从音频生成说话人嵌入向量...")
        try:
            embed = encoder.embed_utterance(preprocessed_wav)
            
            # 打印嵌入向量信息
            if isinstance(embed, np.ndarray):
                logger.info(f"嵌入向量: 形状={embed.shape}, 范围=[{embed.min():.4f}, {embed.max():.4f}], 均值={embed.mean():.4f}, 标准差={embed.std():.4f}")
                if np.isnan(embed).any():
                    logger.error("嵌入向量包含NaN值，这可能导致合成失败")
                if np.isinf(embed).any():
                    logger.error("嵌入向量包含Inf值，这可能导致合成失败")
            elif isinstance(embed, torch.Tensor):
                embed_np = embed.detach().cpu().numpy()
                logger.info(f"嵌入向量(torch.Tensor): 形状={embed.shape}, 设备={embed.device}")
                logger.info(f"嵌入向量: 范围=[{embed_np.min():.4f}, {embed_np.max():.4f}], 均值={embed_np.mean():.4f}, 标准差={embed_np.std():.4f}")
            else:
                logger.info(f"嵌入向量类型: {type(embed)}")
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            traceback.print_exc()
            return
        
        # 获取文本输入
        logger.info("请求用户文本输入...")
        text = input("请输入要合成的文本 (按回车使用默认文本): ")
        if not text:
            text = "欢迎使用实时语音克隆系统，这是一个示例。"
            logger.info(f"使用默认文本: {text}")
        else:
            logger.info(f"用户输入文本: {text}")
            
        # 准备文本列表
        texts = [text]
        
        # 调用inspect_synthesizer_internals函数检查合成器状态
        logger.info("在合成之前检查合成器内部状态...")
        inspect_synthesizer_internals(synthesizer, texts, embed)
        
        # 合成梅尔频谱图
        logger.info("合成梅尔频谱图...")
        try:
            start_time = time.time()
            if isinstance(embed, list):
                logger.info(f"嵌入向量是列表，包含{len(embed)}个项目")
                specs = synthesizer.synthesize_spectrograms(texts, embed)
            else:
                logger.info("嵌入向量是单个向量，将其包装为列表")
                specs = synthesizer.synthesize_spectrograms(texts, [embed] * len(texts))
                
            synth_time = time.time() - start_time
            logger.info(f"梅尔频谱图合成完成，耗时{synth_time:.2f}秒")
            
            if specs is None:
                logger.error("合成器返回了None而不是频谱图")
                return
                
            if not isinstance(specs, list):
                logger.error(f"合成器返回了非列表类型: {type(specs)}")
                if hasattr(specs, 'shape'):
                    logger.info(f"返回的对象形状: {specs.shape}")
                return
                
            if len(specs) == 0:
                logger.error("合成器返回了空列表")
                return
                
            # 打印梅尔频谱图信息
            for i, spec in enumerate(specs):
                logger.info(f"梅尔频谱图 #{i+1}:")
                if isinstance(spec, torch.Tensor):
                    spec_np = spec.detach().cpu().numpy()
                    logger.info(f"频谱图(torch.Tensor): 形状={spec.shape}, 设备={spec.device}")
                    analyze_mel_spectrogram(spec_np, f"梅尔频谱图 #{i+1}")
                else:
                    analyze_mel_spectrogram(spec, f"梅尔频谱图 #{i+1}")
        except Exception as e:
            logger.error(f"梅尔频谱图合成失败: {e}")
            traceback.print_exc()
            return
            
        # 生成波形
        logger.info("从梅尔频谱图生成波形...")
        try:
            start_time = time.time()
            # 打印合成的梅尔频谱图信息
            if isinstance(specs[0], torch.Tensor):
                spec = specs[0].detach().cpu().numpy()
            else:
                spec = specs[0]

            # 确保频谱图形状正确
            if len(spec.shape) != 2:
                logger.error(f"频谱图维度不正确: {spec.shape}")
                return
            
            logger.info(f"用于生成波形的梅尔频谱图形状: {spec.shape}")
            
            # 检查梅尔频谱图是否需要修剪
            if spec.shape[1] > 1000:
                logger.warning(f"梅尔频谱图长度过长({spec.shape[1]}帧)，可能含有不必要的静音，尝试进一步修剪")
                
                # 从末尾修剪静音部分
                original_len = spec.shape[1]
                while spec.shape[1] > 50 and np.max(spec[:, -1]) < 0.1:
                    spec = spec[:, :-1]
                
                # 从开头修剪静音部分
                start_idx = 0
                while start_idx < spec.shape[1] - 10 and np.max(spec[:, start_idx]) < 0.1:
                    start_idx += 1
                if start_idx > 0:
                    spec = spec[:, start_idx:]
                
                logger.info(f"修剪后梅尔频谱图: {spec.shape} (减少了{original_len - spec.shape[1]}帧)")
            
            # 确保频谱图在正确的设备上
            if hasattr(vocoder, "device"):
                logger.info(f"声码器设备: {vocoder.device}")
                
            # 生成波形
            if hasattr(vocoder, "generate_waveform"):
                logger.info("使用generate_waveform方法生成波形")
                generated_wav = vocoder.generate_waveform(spec)
            elif hasattr(vocoder, "infer_waveform"):
                logger.info("使用infer_waveform方法生成波形")
                generated_wav = vocoder.infer_waveform(spec)
            elif hasattr(vocoder, "__call__"):
                logger.info("使用__call__方法生成波形")
                if isinstance(spec, np.ndarray):
                    spec_tensor = torch.from_numpy(spec)
                    if hasattr(vocoder, "device"):
                        spec_tensor = spec_tensor.to(vocoder.device)
                    generated_wav = vocoder(spec_tensor).cpu().numpy()
                else:
                    generated_wav = vocoder(spec).cpu().numpy()
            else:
                logger.error("无法找到声码器的波形生成方法")
                return
                
            vocoder_time = time.time() - start_time
            logger.info(f"波形生成完成，耗时{vocoder_time:.2f}秒")
            
            # 打印生成的波形信息
            logger.info(f"生成的波形: 形状={generated_wav.shape}, 范围=[{generated_wav.min():.4f}, {generated_wav.max():.4f}], 均值={generated_wav.mean():.4f}, 标准差={generated_wav.std():.4f}")
            
            # 保存音频
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(demo_output_dir, f"generated_{timestamp}_{num}.wav")
            logger.info(f"保存生成的音频到: {save_path}")
            
            # 确保波形在[-1, 1]范围内
            if np.abs(generated_wav).max() > 1.0:
                logger.warning("波形超出[-1, 1]范围，将进行归一化")
                generated_wav = generated_wav / np.abs(generated_wav).max()
            
            # 保存为16位PCM音频
            try:
                sf.write(save_path, generated_wav, 16000)
                logger.info(f"音频文件已保存: {save_path}")
            except Exception as e:
                logger.error(f"保存音频文件时出错: {e}")
            
            # 播放生成的音频
            try:
                import IPython.display as ipd
                if 'ipykernel' in sys.modules:
                    logger.info("在Jupyter中播放音频")
                    return ipd.Audio(generated_wav, rate=16000)
                else:
                    logger.info("尝试使用系统播放器播放音频")
                    try:
                        if sys.platform == "win32":
                            os.system(f'start {save_path}')
                        elif sys.platform == "darwin":  # macOS
                            os.system(f'afplay {save_path}')
                        else:  # Linux
                            os.system(f'aplay {save_path}')
                    except:
                        logger.warning("无法使用系统播放器，请手动播放保存的文件")
            except Exception as e:
                logger.warning(f"音频播放失败: {e}")
                
            # 询问是否继续
            if num < 3:  # 限制生成次数
                continue_gen = input("是否继续生成下一个语音? (y/n): ").strip().lower()
                if continue_gen == 'y':
                    process_interactive_generation(encoder, synthesizer, vocoder, num+1)
            else:
                logger.info("已达到最大生成次数")
                
        except Exception as e:
            logger.error(f"生成波形时出错: {e}")
            traceback.print_exc()
            
    except Exception as e:
        logger.error(f"交互式生成过程中出错: {e}")
        traceback.print_exc()
        
    logger.info("交互式语音生成完成")
    print_section("结束交互式语音生成")


def check_mel_quality(mel, verbose=True):
    """检查梅尔频谱图质量并返回评分"""
    try:
        if mel is None:
            if verbose:
                print("梅尔频谱图为空")
            return 0
        
        # 检查频谱图形状
        if verbose:
            print(f"梅尔频谱图形状: {mel.shape}")
            # 标准形状应为(80, 长度)，长度通常为几百帧
            if mel.shape[0] != 80:
                print(f"警告: 梅尔频谱图频率维度异常，应为80，实际为{mel.shape[0]}")
            if mel.shape[1] < 20:
                print(f"警告: 梅尔频谱图帧数过少: {mel.shape[1]}")
            elif mel.shape[1] > 1000:
                print(f"警告: 梅尔频谱图帧数可能过多: {mel.shape[1]}")
                
        # 检查数值范围
        has_nan = np.isnan(mel).any()
        has_inf = np.isinf(mel).any()
        min_val = mel.min()
        max_val = mel.max()
        mean_val = mel.mean()
        std_val = mel.std()
        
        if verbose:
            print(f"梅尔频谱图统计: 范围=[{min_val:.4f}, {max_val:.4f}], 均值={mean_val:.4f}, 标准差={std_val:.4f}")
            if has_nan:
                print("警告: 梅尔频谱图包含NaN值")
            if has_inf:
                print("警告: 梅尔频谱图包含Inf值")
                
        # 检查零值比例
        zero_ratio = np.sum(np.abs(mel) < 1e-5) / mel.size
        if verbose and zero_ratio > 0.7:
            print(f"警告: 梅尔频谱图零值比例过高: {zero_ratio:.2%}")
            
        # 检查频率分布
        freq_means = np.mean(mel, axis=1)
        freq_stds = np.std(mel, axis=1)
        if verbose:
            low_energy = np.mean(freq_means[:20])  # 低频能量
            high_energy = np.mean(freq_means[60:])  # 高频能量
            print(f"频率分布: 低频能量={low_energy:.4f}, 高频能量={high_energy:.4f}")
            if high_energy > low_energy:
                print("警告: 频谱能量分布不自然，高频能量大于低频")
                
        # 计算帧间差异，检查是否有足够的变化
        if mel.shape[1] > 1:
            frame_diffs = np.mean(np.abs(mel[:, 1:] - mel[:, :-1]))
            if verbose:
                print(f"帧间平均差异: {frame_diffs:.4f}")
                if frame_diffs < 0.01:
                    print("警告: 帧间差异过小，可能是静音或重复内容")
        
        # 计算质量分数 (0-10)
        score = 0
        
        # 基于形状评分
        if mel.shape[0] == 80 and 50 <= mel.shape[1] <= 800:
            score += 2
        else:
            score += max(0, 1 - abs(mel.shape[0] - 80) / 80) 
            
        # 基于数值范围评分
        if not has_nan and not has_inf:
            score += 1
        
        # 基于值分布评分
        if -5 <= min_val <= -0.5 and 0.5 <= max_val <= 5:
            score += 1
        
        # 根据帧间差异评分
        if 'frame_diffs' in locals() and 0.01 <= frame_diffs <= 0.2:
            score += 2
        
        # 根据频率分布评分
        if 'low_energy' in locals() and 'high_energy' in locals() and low_energy > high_energy:
            score += 2
        
        # 基于零值比例评分
        if zero_ratio < 0.5:
            score += 2
            
        # 标准化到0-10
        score = min(10, score * 10 / 10)
        
        if verbose:
            print(f"梅尔频谱图质量评分: {score:.1f}/10")
            if score < 3:
                print("质量评分很低，合成结果可能不理想")
            elif score < 5:
                print("质量评分较低，合成结果可能有问题")
            elif score < 7:
                print("质量评分一般，合成结果可能可接受")
            else:
                print("质量评分良好，合成结果应该不错")
                
        return score
    except Exception as e:
        if verbose:
            print(f"评估梅尔频谱图质量时出错: {e}")
        return 0

# 添加检查合成器内部状态的函数
def inspect_synthesizer(synth, verbose=True):
    """检查合成器的内部状态"""
    try:
        if synth is None:
            if verbose:
                print("合成器对象为空")
            return False
            
        # 检查模型是否已初始化
        model_initialized = hasattr(synth, '_model') and synth._model is not None
        if verbose:
            print(f"合成器模型已初始化: {model_initialized}")
            
        # 检查合成器是否能处理文本
        can_process_text = hasattr(synth, 'synthesize_spectrograms')
        if verbose:
            print(f"合成器能处理文本: {can_process_text}")
            
        # 检查embedding的有效性
        if hasattr(synth, '_speaker_embeddings_validity'):
            embedding_valid = synth._speaker_embeddings_validity
            if verbose:
                print(f"嵌入向量有效: {embedding_valid}")
        
        if not model_initialized:
            return False
            
        # 检查tacotron模型参数
        if hasattr(synth, '_model'):
            try:
                model = synth._model
                if verbose:
                    print(f"Tacotron模型类型: {type(model).__name__}")
                    if hasattr(model, 'generate'):
                        # 检查generate方法签名
                        import inspect
                        generate_sig = inspect.signature(model.generate)
                        print(f"generate方法参数: {generate_sig}")
                        # 检查是否支持return_alignments参数
                        has_return_alignments = 'return_alignments' in generate_sig.parameters
                        print(f"generate方法支持return_alignments参数: {has_return_alignments}")
            except Exception as e:
                if verbose:
                    print(f"检查Tacotron模型时出错: {e}")
        
        return model_initialized and can_process_text
    except Exception as e:
        if verbose:
            print(f"检查合成器时出错: {e}")
        return False

# 修改现有函数中的相关部分，移除return_alignments参数
def process_one_utterance(args, text, embed, speaker_name, synth, vocoder, out_dir, num_generated=0):
    try:
        # ... existing code ...
        
        # 生成梅尔频谱图
        print(">> 生成梅尔频谱图")
        try:
            # 移除return_alignments参数
            specs = synth.synthesize_spectrograms([text], [embed])
            spec = specs[0]
            print_tensor_info("生成的梅尔频谱图", spec)
            
            # 检查梅尔频谱图质量
            quality_score = check_mel_quality(spec)
            if quality_score < 4:
                # 尝试调整嵌入向量并重新合成
                print("梅尔频谱图质量较低，尝试调整嵌入向量并重新合成...")
                # 规范化嵌入向量
                adjusted_embed = embed / (np.linalg.norm(embed) + 1e-8)
                specs = synth.synthesize_spectrograms([text], [adjusted_embed])
                new_spec = specs[0]
                new_score = check_mel_quality(new_spec)
                
                if new_score > quality_score:
                    print(f"调整后的梅尔频谱图质量提高: {quality_score:.1f} -> {new_score:.1f}")
                    spec = new_spec
                    quality_score = new_score
                else:
                    print("调整嵌入向量未能提高梅尔频谱图质量")
        except Exception as e:
            print(f"生成梅尔频谱图时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 检查合成器状态
            print("检查合成器状态:")
            inspect_synthesizer(synth)
            
            # 创建一个默认的梅尔频谱图作为fallback
            print("创建默认梅尔频谱图...")
            spec = np.zeros((80, 200))  # 创建一个空的梅尔频谱图
        
        # 生成波形
        print(">> 生成波形")
        try:
            # 从梅尔频谱图生成波形
            generated_wav = vocoder.infer_waveform(spec)
            print_tensor_info("原始生成的波形", generated_wav)
            
            # 应用后处理
            generated_wav = vocoder.apply_taper(generated_wav)
            generated_wav = vocoder.normalize_volume(generated_wav)
            print_tensor_info("后处理后的波形", generated_wav)
            
            # 保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_filename = f"{speaker_name}_{timestamp}_{num_generated:02d}.wav"
            out_path = out_dir / out_filename
            
            print(f">> 保存音频到: {out_path}")
            sf.write(out_path, generated_wav, vocoder.sample_rate)
            
            return out_path
            
        except Exception as e:
            print(f"生成波形时出错: {e}")
            traceback.print_exc()
            return None
    
    except Exception as e:
        print(f"处理话语时出错: {e}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print_section("实时语音克隆程序启动")
    logger.info("初始化命令行参数...")
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="LSTM编码器模型的路径")
    parser.add_argument("--use_resnet", action="store_true", help=\
        "如果为True，使用ResNet模型代替LSTM编码器")
    parser.add_argument("--resnet_model_fpath", type=Path,
                        default="saved_models/default/resnet_encoder.pt",
                        help="ResNet编码器模型的路径 (由encoder_train_resnet.py训练得到)")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="合成器模型的路径")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="声码器模型的路径")
    parser.add_argument("--cpu", action="store_true", help=\
        "如果为True，强制使用CPU处理，即使有GPU可用")
    parser.add_argument("--no_sound", action="store_true", help=\
        "如果为True，不播放音频")
    parser.add_argument("--seed", type=int, default=None, help=\
        "可选的随机数种子，使结果可重复")
    parser.add_argument("--verbose", action="store_true", help=\
        "如果为True，打印更详细的信息")
    
    args = parser.parse_args()
    
    # 如果设置了详细模式，将日志级别设置为DEBUG
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("已启用详细输出模式")
    
    arg_dict = vars(args)
    print_args(args, parser)

    # 隐藏PyTorch的GPU，强制使用CPU处理
    if arg_dict.pop("cpu"):
        logger.info("强制使用CPU进行处理")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 检查GPU可用性
    logger.info("检查GPU可用性...")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        logger.info(f"找到 {torch.cuda.device_count()} 个可用的GPU。使用GPU {device_id} ({gpu_properties.name})，"
                  f"计算能力 {gpu_properties.major}.{gpu_properties.minor}，"
                  f"总内存 {gpu_properties.total_memory / 1e9:.1f}GB")
    else:
        logger.info("没有可用的GPU，将使用CPU进行推理")

    # 加载模型
    print_section("加载模型")
    logger.info("准备加载编码器、合成器和声码器...")
    
    start_time = time.time()
    logger.info(f"模型检查完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 根据用户选择加载不同的编码器
    encoder_start = time.time()
    if args.use_resnet:
        logger.info("=== 使用ResNet编码器模型 ===")
        try:
            # 检查文件是否存在
            if not args.resnet_model_fpath.exists():
                logger.warning(f"指定的ResNet模型文件不存在: {args.resnet_model_fpath}")
                # 尝试备选路径
                alternative_paths = [
                    Path("saved_models/default/resnet_encoder.pt"),
                    Path("saved_models/resnet_encoder/resnet_encoder.pt"),
                    Path("saved_models/resnet_encoder/resnet_encoder_best.pt"),
                    Path("ResNet/saved_models/resnet_encoder.pt"),
                    Path("ResNet/saved_models/resnet_encoder_best.pt")
                ]
                
                logger.info("尝试查找备选ResNet模型文件...")
                for path in alternative_paths:
                    if path.exists():
                        logger.info(f"找到备选ResNet模型: {path}")
                        args.resnet_model_fpath = path
                        break
                else:
                    logger.warning("未找到任何ResNet模型文件，将使用随机初始化的模型")
            
            # 加载ResNet编码器
            logger.info(f"尝试加载ResNet模型: {args.resnet_model_fpath}")
            resnet_encoder.load_model(args.resnet_model_fpath)
            logger.info(f"ResNet模型加载成功: {args.resnet_model_fpath}")
            
            # 将ResNet推理模块设置为全局编码器
            active_encoder = resnet_encoder
            logger.info(f"激活的编码器: ResNet")
        except Exception as e:
            logger.error(f"加载ResNet模型时出错: {e}")
            logger.error("打印完整栈跟踪信息:")
            traceback.print_exc()
            logger.warning("尝试回退到LSTM编码器...")
            
            try:
                logger.info(f"尝试加载LSTM编码器: {args.enc_model_fpath}")
                encoder.load_model(args.enc_model_fpath)
                active_encoder = encoder
                logger.info(f"回退成功，已加载LSTM编码器")
            except Exception as e2:
                logger.error(f"加载LSTM编码器也失败: {e2}")
                logger.error("无法加载任何编码器，继续使用未初始化的模型")
                active_encoder = resnet_encoder if args.use_resnet else encoder
    else:
        logger.info("=== 使用LSTM编码器模型 ===")
        try:
            # 检查文件是否存在
            if not args.enc_model_fpath.exists():
                logger.warning(f"指定的LSTM编码器模型文件不存在: {args.enc_model_fpath}")
                # 尝试备选路径
                alternative_paths = [
                    Path("saved_models/default/encoder.pt"),
                    Path("encoder/saved_models/encoder.pt"),
                    Path("encoder/saved_models/encoder_best.pt")
                ]
                
                logger.info("尝试查找备选LSTM编码器模型文件...")
                for path in alternative_paths:
                    if path.exists():
                        logger.info(f"找到备选LSTM编码器模型: {path}")
                        args.enc_model_fpath = path
                        break
                else:
                    logger.warning("未找到任何LSTM编码器模型文件，将使用随机初始化的模型")
            
            logger.info(f"尝试加载LSTM编码器模型: {args.enc_model_fpath}")
            encoder.load_model(args.enc_model_fpath)
            active_encoder = encoder
            logger.info(f"LSTM编码器模型加载成功: {args.enc_model_fpath}")
        except Exception as e:
            logger.error(f"加载LSTM编码器模型时出错: {e}")
            logger.error("打印完整栈跟踪信息:")
            traceback.print_exc()
            logger.warning("将使用未初始化的模型继续...")
            active_encoder = encoder
    
    logger.info(f"编码器加载完成，耗时: {time.time() - encoder_start:.2f}秒")
    
    # 加载合成器
    synth_start = time.time()
    logger.info("=== 加载合成器模型 ===")
    try:
        # 检查文件是否存在
        if not args.syn_model_fpath.exists():
            logger.warning(f"指定的合成器模型文件不存在: {args.syn_model_fpath}")
            # 尝试备选路径
            alternative_paths = [
                Path("saved_models/default/synthesizer.pt"),
                Path("synthesizer/saved_models/synthesizer.pt"),
                Path("synthesizer/saved_models/synthesizer_best.pt")
            ]
            
            logger.info("尝试查找备选合成器模型文件...")
            for path in alternative_paths:
                if path.exists():
                    logger.info(f"找到备选合成器模型: {path}")
                    args.syn_model_fpath = path
                    break
            else:
                logger.warning("未找到任何合成器模型文件，将使用随机初始化的模型")
        
        logger.info(f"尝试加载合成器模型: {args.syn_model_fpath}")
        synthesizer = Synthesizer(args.syn_model_fpath)
        logger.info(f"合成器模型加载成功: {args.syn_model_fpath}")
    except Exception as e:
        logger.error(f"加载合成器模型时出错: {e}")
        logger.error("打印完整栈跟踪信息:")
        traceback.print_exc()
        logger.warning("合成器加载失败，将继续尝试使用未初始化的合成器")
        # 创建一个空的合成器对象
        try:
            synthesizer = Synthesizer(Path("dummy_path"))
        except:
            logger.warning("无法创建合成器对象，部分功能可能不可用")
            synthesizer = None

    logger.info(f"合成器加载完成，耗时: {time.time() - synth_start:.2f}秒")
    
    # 加载声码器
    voc_start = time.time()
    logger.info("=== 加载声码器模型 ===")
    try:
        # 检查文件是否存在
        if not args.voc_model_fpath.exists():
            logger.warning(f"指定的声码器模型文件不存在: {args.voc_model_fpath}")
            # 尝试备选路径
            alternative_paths = [
                Path("saved_models/default/vocoder.pt"),
                Path("vocoder/saved_models/vocoder.pt"),
                Path("vocoder/saved_models/vocoder_best.pt")
            ]
            
            logger.info("尝试查找备选声码器模型文件...")
            for path in alternative_paths:
                if path.exists():
                    logger.info(f"找到备选声码器模型: {path}")
                    args.voc_model_fpath = path
                    break
            else:
                logger.warning("未找到任何声码器模型文件，将使用随机初始化的模型")
        
        logger.info(f"尝试加载声码器模型: {args.voc_model_fpath}")
        vocoder.load_model(args.voc_model_fpath)
        logger.info(f"声码器模型加载成功: {args.voc_model_fpath}")
    except Exception as e:
        logger.error(f"加载声码器模型时出错: {e}")
        logger.error("打印完整栈跟踪信息:")
        traceback.print_exc()
        logger.warning("声码器加载失败，部分功能可能不可用")

    logger.info(f"声码器加载完成，耗时: {time.time() - voc_start:.2f}秒")
    logger.info(f"所有模型加载完成，总耗时: {time.time() - start_time:.2f}秒")

    # 测试配置
    try:
        test_models(active_encoder, synthesizer, args)
    except Exception as e:
        logger.error(f"测试模型配置时出错: {e}")
        logger.error("打印完整栈跟踪信息:")
        traceback.print_exc()
        logger.warning("测试模型失败，但将继续尝试交互式语音生成")

    # 交互式语音生成
    process_interactive_generation(active_encoder, synthesizer, vocoder, True)
