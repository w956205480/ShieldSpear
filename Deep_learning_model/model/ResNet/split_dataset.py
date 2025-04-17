"""
# 说话人验证数据集划分脚本
#
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


# 添加项目根目录到系统路径，以便导入相关模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoder.params_data import *

# 全局变量定义
test_dirs = []  # 用于存储测试集目录


def create_sources_file(speaker_dir, utterances):
    """
    为说话人目录创建_sources.txt文件
    
    参数：
        speaker_dir: 说话人目录路径
        utterances: 语音文件列表
    """
    sources_path = speaker_dir / "_sources.txt"
    
    with sources_path.open("w") as f:
        for utterance in utterances:
            # 提取文件名（不含扩展名）
            basename = utterance.stem
            
            # 如果是预处理过的npy文件，源音频可能是wav或mp3
            possible_sources = [
                str(utterance.with_suffix(".wav").absolute()),
                str(utterance.with_suffix(".mp3").absolute()),
                f"[未知来源]{basename}"  # 如果找不到源文件，添加占位符
            ]
            
            # 尝试找到存在的源文件
            source_path = next((s for s in possible_sources[:2] if Path(s).exists()), possible_sources[2])
            
            # 将文件名和源路径写入_sources.txt
            f.write(f"{utterance.name},{source_path}\n")


def add_to_sources_file(main_dir, speaker_id, utterances):
    """
    将说话人的语音片段信息添加到主目录的_sources.txt文件中
    
    参数：
        main_dir: 主目录路径（train或test目录）
        speaker_id: 说话人ID
        utterances: 语音文件列表
    """
    sources_path = main_dir / "_sources.txt"
    
    # 使用追加模式，保留已有内容
    with sources_path.open("a") as f:
        for utterance in utterances:
            # 提取文件名（不含扩展名）
            basename = utterance.stem
            
            # 如果是预处理过的npy文件，源音频可能是wav或mp3
            possible_sources = [
                str(utterance.with_suffix(".wav").absolute()),
                str(utterance.with_suffix(".mp3").absolute()),
                f"[未知来源]{basename}"  # 如果找不到源文件，添加占位符
            ]
            
            # 尝试找到存在的源文件
            source_path = next((s for s in possible_sources[:2] if Path(s).exists()), possible_sources[2])
            
            # 将文件名和源路径写入_sources.txt，加入说话人ID信息
            f.write(f"{speaker_id}/{utterance.name},{source_path}\n")


def split_speaker_dataset(dataset_root, output_root=None, train_ratio=0.9, min_utterances=10, sv2tts_structure=True):
    """
    将说话人数据集划分为训练集和测试集
    
    参数：
        dataset_root: 原始数据集根目录
        output_root: 输出目录（如果为None，则使用dataset_root）
        train_ratio: 训练集比例
        min_utterances: 每个说话人的最少语音片段数
        sv2tts_structure: 是否创建SV2TTS目录结构（dataset/SV2TTS/encoder/train和test）
    """
    if not isinstance(dataset_root, Path):
        dataset_root = Path(dataset_root)
    
    if output_root is None:
        output_root = dataset_root
    elif not isinstance(output_root, Path):
        output_root = Path(output_root)
    
    # 创建输出目录，根据是否需要SV2TTS结构
    if sv2tts_structure:
        # 这里确保创建dataset/SV2TTS/encoder目录结构
        # 如果output_root不是绝对路径，使之成为相对于当前目录的路径
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root
        
        # 检查output_root是否已经包含dataset路径
        if output_root.name != "dataset" and not (output_root / "dataset").exists():
            sv2tts_dir = output_root / "dataset" / "SV2TTS"
        else:
            sv2tts_dir = output_root / "SV2TTS"
        
        encoder_dir = sv2tts_dir / "encoder"
        train_dir = encoder_dir / "train"
        test_dir = encoder_dir / "test"
        
        # 输出真实路径信息
        print(f"创建SV2TTS目录结构:")
        print(f"  SV2TTS目录: {sv2tts_dir.absolute()}")
        print(f"  Encoder目录: {encoder_dir.absolute()}")
        print(f"  训练集目录: {train_dir.absolute()}")
        print(f"  测试集目录: {test_dir.absolute()}")
    else:
        train_dir = output_root / "train"
        test_dir = output_root / "test"
    
    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # 如果是SV2TTS结构，在train和test目录创建_sources.txt文件
    if sv2tts_structure:
        # 创建encoder目录下的全局_sources.txt文件
        encoder_sources = train_dir.parent / "_sources.txt"
        with open(encoder_sources, "w") as f:
            f.write("")
        print(f"创建文件: {encoder_sources.absolute()}")
        
        # 确保train和test目录中也有一个空的_sources.txt
        train_sources = train_dir / "_sources.txt"
        with open(train_sources, "w") as f:
            f.write("")
        print(f"创建文件: {train_sources.absolute()}")
        
        test_sources = test_dir / "_sources.txt"
        with open(test_sources, "w") as f:
            f.write("")
        print(f"创建文件: {test_sources.absolute()}")
        
        # 验证文件是否成功创建
        if not encoder_sources.exists():
            print(f"警告: 文件创建失败 - {encoder_sources.absolute()}")
        if not train_sources.exists():
            print(f"警告: 文件创建失败 - {train_sources.absolute()}")
        if not test_sources.exists():
            print(f"警告: 文件创建失败 - {test_sources.absolute()}")
            
        # 确认这些文件对应的相对路径是否和错误消息中的匹配
        relative_test_path = Path("dataset/SV2TTS/encoder/test/_sources.txt")
        if test_sources.absolute().as_posix().endswith(relative_test_path.as_posix()):
            print(f"√ 测试集_sources.txt路径匹配预期的相对路径: {relative_test_path}")
        else:
            print(f"! 警告: 测试集_sources.txt路径与预期不匹配")
            print(f"  实际: {test_sources.absolute()}")
            print(f"  预期: 路径应以 {relative_test_path} 结尾")
            print(f"  建议: 请尝试使用 --output_root . 选项，或调整输出路径")
    
    # 获取所有说话人的目录
    speakers = [d for d in dataset_root.glob("*") if d.is_dir()]
    print(f"共找到 {len(speakers)} 个说话人目录")
    
    # 统计信息
    total_utterances = 0
    train_utterances = 0
    test_utterances = 0
    skipped_speakers = 0
    
    # 处理每个说话人
    for speaker_dir in tqdm(speakers, desc="处理说话人目录"):
        speaker_id = speaker_dir.name
        
        # 获取该说话人的所有语音片段(使用npy文件或者wav文件）
        utterances = list(speaker_dir.glob("*.npy"))
        if len(utterances) == 0:
            utterances = list(speaker_dir.glob("*.wav"))
        
        # 如果语音片段太少，则跳过该说话人
        if len(utterances) < min_utterances:
            skipped_speakers += 1
            continue
        
        # 随机打乱语音片段
        random.shuffle(utterances)
        
        # 划分训练集和测试集
        train_count = int(len(utterances) * train_ratio)
        train_utterances += train_count
        test_utterances += (len(utterances) - train_count)
        total_utterances += len(utterances)
        
        # 创建说话人目录
        train_speaker_dir = train_dir / speaker_id
        test_speaker_dir = test_dir / speaker_id
        train_speaker_dir.mkdir(exist_ok=True)
        test_speaker_dir.mkdir(exist_ok=True)
        
        # 分别处理训练集和测试集
        train_files = []
        test_files = []
        
        # 复制或链接文件
        for i, utterance in enumerate(utterances):
            if i < train_count:
                target_dir = train_speaker_dir
                target_path = target_dir / utterance.name
                train_files.append(target_path)
            else:
                target_dir = test_speaker_dir
                target_path = target_dir / utterance.name
                test_files.append(target_path)
            
            # 创建符号链接而不是复制文件，节省空间
            if not target_path.exists():
                try:
                    # 如果在Windows上可能会有权限问题，则使用复制而不是符号链接
                    if os.name == 'nt':
                        import shutil
                        shutil.copy2(str(utterance), str(target_path))
                    else:
                        os.symlink(str(utterance.absolute()), str(target_path))
                except Exception as e:
                    print(f"创建链接失败，尝试复制文件: {e}")
                    import shutil
                    shutil.copy2(str(utterance), str(target_path))
        
        # 将说话人的语音片段信息添加到主目录的_sources.txt文件中
        if train_files:
            add_to_sources_file(train_dir, speaker_id, train_files)
        if test_files:
            add_to_sources_file(test_dir, speaker_id, test_files)
    
    # 打印统计信息
    print(f"\n数据集划分完成:")
    print(f"  总说话人数: {len(speakers)}")
    print(f"  有效说话人数: {len(speakers) - skipped_speakers}")
    print(f"  跳过的说话人数 (语音片段少于{min_utterances}): {skipped_speakers}")
    print(f"  总语音片段数: {total_utterances}")
    print(f"  训练集语音片段数: {train_utterances} ({train_utterances/total_utterances*100:.2f}%)")
    print(f"  测试集语音片段数: {test_utterances} ({test_utterances/total_utterances*100:.2f}%)")
    print(f"  训练集路径: {train_dir}")
    print(f"  测试集路径: {test_dir}")
    
    # 返回创建的目录
    return train_dir.parent if sv2tts_structure else output_root


def split_by_speakers(dataset_root, output_root=None, test_speakers_ratio=0.1, min_utterances=10, sv2tts_structure=True):
    """
    按说话人划分数据集，某些说话人完全进入测试集，其他进入训练集
    
    参数：
        dataset_root: 原始数据集根目录
        output_root: 输出目录（如果为None，则使用dataset_root）
        test_speakers_ratio: 测试集中的说话人比例
        min_utterances: 每个说话人的最少语音片段数
        sv2tts_structure: 是否创建SV2TTS目录结构（dataset/SV2TTS/encoder/train和test）
    """
    if not isinstance(dataset_root, Path):
        dataset_root = Path(dataset_root)
    
    if output_root is None:
        output_root = dataset_root
    elif not isinstance(output_root, Path):
        output_root = Path(output_root)
    
    # 创建输出目录，根据是否需要SV2TTS结构
    if sv2tts_structure:
        # 这里确保创建dataset/SV2TTS/encoder目录结构
        # 如果output_root不是绝对路径，使之成为相对于当前目录的路径
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root
        
        # 检查output_root是否已经包含dataset路径
        if output_root.name != "dataset" and not (output_root / "dataset").exists():
            sv2tts_dir = output_root / "dataset" / "SV2TTS"
        else:
            sv2tts_dir = output_root / "SV2TTS"
        
        encoder_dir = sv2tts_dir / "encoder"
        train_dir = encoder_dir / "train"
        test_dir = encoder_dir / "test"
        
        # 输出真实路径信息
        print(f"创建SV2TTS目录结构:")
        print(f"  SV2TTS目录: {sv2tts_dir.absolute()}")
        print(f"  Encoder目录: {encoder_dir.absolute()}")
        print(f"  训练集目录: {train_dir.absolute()}")
        print(f"  测试集目录: {test_dir.absolute()}")
    else:
        train_dir = output_root / "train"
        test_dir = output_root / "test"
    
    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # 如果是SV2TTS结构，在train和test目录创建_sources.txt文件
    if sv2tts_structure:
        # 创建encoder目录下的全局_sources.txt文件
        encoder_sources = train_dir.parent / "_sources.txt"
        with open(encoder_sources, "w") as f:
            f.write("")
        print(f"创建文件: {encoder_sources.absolute()}")
        
        # 确保train和test目录中也有一个空的_sources.txt
        train_sources = train_dir / "_sources.txt"
        with open(train_sources, "w") as f:
            f.write("")
        print(f"创建文件: {train_sources.absolute()}")
        
        test_sources = test_dir / "_sources.txt"
        with open(test_sources, "w") as f:
            f.write("")
        print(f"创建文件: {test_sources.absolute()}")
        
        # 验证文件是否成功创建
        if not encoder_sources.exists():
            print(f"警告: 文件创建失败 - {encoder_sources.absolute()}")
        if not train_sources.exists():
            print(f"警告: 文件创建失败 - {train_sources.absolute()}")
        if not test_sources.exists():
            print(f"警告: 文件创建失败 - {test_sources.absolute()}")
            
        # 确认这些文件对应的相对路径是否和错误消息中的匹配
        relative_test_path = Path("dataset/SV2TTS/encoder/test/_sources.txt")
        if test_sources.absolute().as_posix().endswith(relative_test_path.as_posix()):
            print(f"√ 测试集_sources.txt路径匹配预期的相对路径: {relative_test_path}")
        else:
            print(f"! 警告: 测试集_sources.txt路径与预期不匹配")
            print(f"  实际: {test_sources.absolute()}")
            print(f"  预期: 路径应以 {relative_test_path} 结尾")
            print(f"  建议: 请尝试使用 --output_root . 选项，或调整输出路径")
    
    # 获取所有说话人的目录
    speakers = [d for d in dataset_root.glob("*") if d.is_dir()]
    print(f"共找到 {len(speakers)} 个说话人目录")
    
    # 过滤掉语音片段太少的说话人
    qualified_speakers = []
    for speaker_dir in speakers:
        utterances = list(speaker_dir.glob("*.npy"))
        if len(utterances) == 0:
            utterances = list(speaker_dir.glob("*.wav"))
        
        if len(utterances) >= min_utterances:
            qualified_speakers.append((speaker_dir, len(utterances), utterances))
    
    # 按语音片段数量排序
    qualified_speakers.sort(key=lambda x: x[1], reverse=True)
    
    # 随机打乱但保持一定的平衡性（确保测试集中有各种数量的语音片段）
    batched_speakers = []
    batch_size = 10  # 每批次的说话人数量
    for i in range(0, len(qualified_speakers), batch_size):
        batch = qualified_speakers[i:i+batch_size]
        random.shuffle(batch)
        batched_speakers.extend(batch)
    
    # 划分训练集和测试集
    test_speakers_count = int(len(batched_speakers) * test_speakers_ratio)
    test_speakers = batched_speakers[:test_speakers_count]
    train_speakers = batched_speakers[test_speakers_count:]
    
    # 统计信息
    train_utterances = sum(count for _, count, _ in train_speakers)
    test_utterances = sum(count for _, count, _ in test_speakers)
    total_utterances = train_utterances + test_utterances
    
    # 处理测试集说话人
    for speaker_dir, _, utterances in tqdm(test_speakers, desc="处理测试集说话人"):
        speaker_id = speaker_dir.name
        test_speaker_dir = test_dir / speaker_id
        test_speaker_dir.mkdir(exist_ok=True)
        
        test_files = []
        
        # 创建符号链接或复制文件
        for utterance in utterances:
            target_path = test_speaker_dir / utterance.name
            test_files.append(target_path)
            
            if not target_path.exists():
                try:
                    if os.name == 'nt':
                        import shutil
                        shutil.copy2(str(utterance), str(target_path))
                    else:
                        os.symlink(str(utterance.absolute()), str(target_path))
                except Exception as e:
                    print(f"创建链接失败，尝试复制文件: {e}")
                    import shutil
                    shutil.copy2(str(utterance), str(target_path))
        
        # 将说话人的语音片段信息添加到测试集_sources.txt文件中
        add_to_sources_file(test_dir, speaker_id, test_files)
    
    # 处理训练集说话人
    for speaker_dir, _, utterances in tqdm(train_speakers, desc="处理训练集说话人"):
        speaker_id = speaker_dir.name
        train_speaker_dir = train_dir / speaker_id
        train_speaker_dir.mkdir(exist_ok=True)
        
        train_files = []
        
        # 创建符号链接或复制文件
        for utterance in utterances:
            target_path = train_speaker_dir / utterance.name
            train_files.append(target_path)
            
            if not target_path.exists():
                try:
                    if os.name == 'nt':
                        import shutil
                        shutil.copy2(str(utterance), str(target_path))
                    else:
                        os.symlink(str(utterance.absolute()), str(target_path))
                except Exception as e:
                    print(f"创建链接失败，尝试复制文件: {e}")
                    import shutil
                    shutil.copy2(str(utterance), str(target_path))
        
        # 将说话人的语音片段信息添加到训练集_sources.txt文件中
        add_to_sources_file(train_dir, speaker_id, train_files)
    
    # 打印统计信息
    print(f"\n数据集划分完成:")
    print(f"  有效说话人总数: {len(qualified_speakers)}")
    print(f"  训练集说话人数: {len(train_speakers)}")
    print(f"  测试集说话人数: {len(test_speakers)}")
    print(f"  总语音片段数: {total_utterances}")
    print(f"  训练集语音片段数: {train_utterances} ({train_utterances/total_utterances*100:.2f}%)")
    print(f"  测试集语音片段数: {test_utterances} ({test_utterances/total_utterances*100:.2f}%)")
    print(f"  训练集路径: {train_dir}")
    print(f"  测试集路径: {test_dir}")
    
    # 返回创建的目录
    return train_dir.parent if sv2tts_structure else output_root


def process_speaker_directories(all_sub_dirs, train_dir, test_dir, sv2tts_structure=False):
    """处理说话人目录，并生成必要的文件结构"""
    # 创建_sources.txt文件
    if sv2tts_structure:
        # 确保创建目录结构
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保encoder目录中的_sources.txt存在
        encoder_dir = train_dir.parent
        encoder_sources = encoder_dir / "_sources.txt"
        if not encoder_sources.exists():
            with open(encoder_sources, "w", encoding="utf-8") as f:
                f.write("")  # 创建空文件
            print(f"创建编码器目录_sources.txt: {encoder_sources.absolute()}")
        
        # 训练目录的_sources.txt - 清空之前的内容
        train_sources = train_dir / "_sources.txt"
        with open(train_sources, "w", encoding="utf-8") as f:
            f.write("")  # 创建空文件
        print(f"创建训练目录_sources.txt: {train_sources.absolute()}")
        
        # 测试目录的_sources.txt - 清空之前的内容
        test_sources = test_dir / "_sources.txt"
        with open(test_sources, "w", encoding="utf-8") as f:
            f.write("")  # 创建空文件
        print(f"创建测试目录_sources.txt: {test_sources.absolute()}")
    
    # 准备测试集目录列表，用于检查说话人是否属于测试集
    test_speaker_dirs = [d.name for d in test_dir.iterdir() if d.is_dir()]
    
    # 处理说话人目录
    for speaker_dir in tqdm(all_sub_dirs, desc="处理说话人目录"):
        utterances = []
        # 搜索音频文件
        for ext in ["*.npy", "*.wav", "*.flac", "*.mp3"]:
            utterances.extend(list(speaker_dir.glob(ext)))
        
        if len(utterances) == 0:
            continue  # 跳过没有音频文件的目录
        
        # 创建每个说话人的目标目录
        speaker_name = speaker_dir.name
        train_speaker_dir = train_dir / speaker_name
        test_speaker_dir = test_dir / speaker_name
        
        train_speaker_dir.mkdir(exist_ok=True)
        belongs_to_test = speaker_name in test_speaker_dirs
        if belongs_to_test:
            test_speaker_dir.mkdir(exist_ok=True)
        
        # 将所有信息添加到主目录的_sources.txt中
        with open(train_dir / "_sources.txt", "a", encoding="utf-8") as f:
            for utterance in utterances:
                # 使用相对路径格式: speaker_id/utterance.name
                f.write(f"{speaker_name}/{utterance.name},{utterance.absolute()}\n")
        
        if belongs_to_test:
            with open(test_dir / "_sources.txt", "a", encoding="utf-8") as f:
                for utterance in utterances:
                    # 使用相对路径格式: speaker_id/utterance.name
                    f.write(f"{speaker_name}/{utterance.name},{utterance.absolute()}\n")
        
    # 验证创建的文件
    if sv2tts_structure:
        validate_directory_structure(train_dir, test_dir)

def validate_directory_structure(train_dir, test_dir):
    """验证目录结构是否正确"""
    # 验证_sources.txt文件
    encoder_dir = train_dir.parent
    validation_failed = False
    
    files_to_check = [
        (encoder_dir / "_sources.txt", "编码器目录"),
        (train_dir / "_sources.txt", "训练目录"),
        (test_dir / "_sources.txt", "测试目录")
    ]
    
    print("\n验证目录结构...")
    for file_path, desc in files_to_check:
        if not file_path.exists():
            print(f"错误: {desc}的_sources.txt文件不存在: {file_path.absolute()}")
            validation_failed = True
            try:
                # 尝试修复
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")
                print(f"已创建缺失的{desc}_sources.txt文件")
            except Exception as e:
                print(f"无法创建{desc}_sources.txt文件: {e}")
        else:
            print(f"验证通过: {desc}_sources.txt文件存在: {file_path.absolute()}")
    
    # 检查说话人目录
    speaker_dirs_train = [d for d in train_dir.iterdir() if d.is_dir()]
    speaker_dirs_test = [d for d in test_dir.iterdir() if d.is_dir()]
    
    # 验证训练集
    if not speaker_dirs_train:
        print(f"警告: 训练目录中没有说话人目录")
    else:
        print(f"训练目录包含 {len(speaker_dirs_train)} 个说话人")
        # 检查第一个说话人目录和训练目录的_sources.txt
        check_first_speaker(speaker_dirs_train, "训练", train_dir)
    
    # 验证测试集
    if not speaker_dirs_test:
        print(f"警告: 测试目录中没有说话人目录")
    else:
        print(f"测试目录包含 {len(speaker_dirs_test)} 个说话人")
        # 检查第一个说话人目录和测试目录的_sources.txt
        check_first_speaker(speaker_dirs_test, "测试", test_dir)
    
    if validation_failed:
        print("警告: 目录结构验证失败，已尝试修复问题")
    else:
        print("目录结构验证通过!")

def check_first_speaker(speaker_dirs, desc, main_dir):
    """检查第一个说话人目录和主目录的_sources.txt文件"""
    first_speaker = speaker_dirs[0]
    
    # 检查主目录的_sources.txt文件
    main_sources_file = main_dir / "_sources.txt"
    if not main_sources_file.exists():
        print(f"错误: {desc}集目录缺少_sources.txt: {main_sources_file.absolute()}")
        # 创建空的_sources.txt文件
        try:
            with open(main_sources_file, "w", encoding="utf-8") as f:
                f.write("")
            print(f"已为{desc}集目录创建空的_sources.txt文件")
        except Exception as e:
            print(f"无法创建{desc}集目录的_sources.txt文件: {e}")
    else:
        # 验证文件内容
        try:
            with open(main_sources_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                print(f"警告: {desc}集目录的_sources.txt文件为空")
                # 尝试添加第一个说话人的语音片段信息
                utterances = []
                speaker_name = first_speaker.name
                for ext in ["*.npy", "*.wav", "*.flac", "*.mp3"]:
                    utterances.extend(list(first_speaker.glob(ext)))
                
                if utterances:
                    with open(main_sources_file, "a", encoding="utf-8") as f:
                        for utterance in utterances:
                            f.write(f"{speaker_name}/{utterance.name},{utterance.absolute()}\n")
                    print(f"已添加{desc}集第一个说话人的语音片段信息到_sources.txt")
            else:
                print(f"验证通过: {desc}集目录的_sources.txt文件包含 {len(lines)} 行")
        except Exception as e:
            print(f"验证{desc}集目录的_sources.txt文件时出错: {e}")
    
    # 检查第一个说话人目录中是否有音频文件
    utterances = []
    for ext in ["*.npy", "*.wav", "*.flac", "*.mp3"]:
        utterances.extend(list(first_speaker.glob(ext)))
    
    if not utterances:
        print(f"警告: {desc}集第一个说话人目录中没有找到音频文件")
    else:
        print(f"验证通过: {desc}集第一个说话人目录包含 {len(utterances)} 个音频文件")

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='语音数据集分割工具')
    parser.add_argument('dataset_root', type=Path, help='原始数据集目录，包含所有说话人的子目录')
    parser.add_argument('--output_root', type=Path, default=None, help='输出目录，默认为dataset_root/split')
    parser.add_argument('--split_mode', type=str, choices=['utterance', 'speaker'], default='utterance',
                        help='分割模式: utterance-按语音片段划分, speaker-按说话人划分')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='训练集比例 (0.0-1.0)')
    parser.add_argument('--test_speakers_ratio', type=float, default=0.1, help='用于测试的说话人比例 (0.0-1.0)')
    parser.add_argument('--min_utterances', type=int, default=5, help='每个说话人最少的语音片段数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于复现结果')
    parser.add_argument('--sv2tts_structure', action='store_true', help='使用SV2TTS目录结构 (dataset/SV2TTS/encoder/train和test)')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # 检查输入目录是否存在
    if not args.dataset_root.exists():
        print(f"错误: 数据集目录不存在: {args.dataset_root}")
        return
    
    # 设置输出目录
    if args.output_root is None:
        args.output_root = args.dataset_root / "split"
    
    # 确保输出目录为绝对路径
    args.output_root = args.output_root.absolute()
    
    # 为SV2TTS结构设置特定的目录
    if args.sv2tts_structure:
        # 在当前工作目录下创建dataset/SV2TTS结构
        sv2tts_root = args.output_root / "dataset" / "SV2TTS"
        sv2tts_root = sv2tts_root.absolute()  # 确保是绝对路径
        
        encoder_dir = sv2tts_root / "encoder"
        train_dir = encoder_dir / "train"
        test_dir = encoder_dir / "test"
        
        # 创建目录
        for dir_path in [sv2tts_root, encoder_dir, train_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"创建SV2TTS目录结构:")
        print(f"  - 根目录: {sv2tts_root}")
        print(f"  - 编码器: {encoder_dir}")
        print(f"  - 训练集: {train_dir}")
        print(f"  - 测试集: {test_dir}")
        
        # 创建这些目录中的_sources.txt文件
        for dir_path, desc in [(encoder_dir, "编码器"), (train_dir, "训练"), (test_dir, "测试")]:
            sources_file = dir_path / "_sources.txt"
            with open(sources_file, "w", encoding="utf-8") as f:
                f.write("")  # 创建空文件
            print(f"创建{desc}目录的_sources.txt文件: {sources_file.absolute()}")
    else:
        # 使用标准目录结构
        train_dir = args.output_root / "train"
        test_dir = args.output_root / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有子目录（说话人目录）
    all_sub_dirs = [d for d in args.dataset_root.iterdir() if d.is_dir()]
    if not all_sub_dirs:
        print(f"错误: 找不到说话人目录，请检查数据集路径: {args.dataset_root}")
        return
    
    print(f"找到 {len(all_sub_dirs)} 个说话人目录")
    
    # 按照指定的模式分割数据集
    global test_dirs  # 使其在函数内可访问
    
    if args.split_mode == 'speaker':
        # 按说话人划分
        output_dir = split_by_speakers(args.dataset_root, args.output_root, args.test_speakers_ratio, args.min_utterances, args.sv2tts_structure)
    else:
        # 按语音片段划分 (默认)
        output_dir = split_speaker_dataset(args.dataset_root, args.output_root, args.train_ratio, args.min_utterances, args.sv2tts_structure)
    
    # 处理已完成，不需要再次处理
    # process_speaker_directories(all_sub_dirs, train_dir, test_dir, args.sv2tts_structure)
    
    # 输出使用指南
    if args.sv2tts_structure:
        sv2tts_path = args.output_root / "dataset" / "SV2TTS"
        absolute_sv2tts = sv2tts_path.absolute()
        
        print("\n==========================")
        print("处理完成! 使用以下命令训练模型:")
        print(f"python ResNet/train.py <训练ID> {absolute_sv2tts} --resnet_type resnet50")
        print("==========================")
        print(f"重要提示: 如果遇到'找不到dataset/SV2TTS/encoder/test/_sources.txt'错误")
        print(f"确保在正确的工作目录下运行，该目录中应当包含 dataset/SV2TTS 目录")
        print(f"或者使用完整的绝对路径: {absolute_sv2tts}")
        
        # 额外验证
        expected_path = Path('dataset/SV2TTS/encoder/test/_sources.txt')
        actual_path = sv2tts_path / "encoder" / "test" / "_sources.txt"
        
        print("\n路径验证:")
        if actual_path.exists():
            print(f"✓ 已创建: {actual_path.absolute()}")
            
            if expected_path.exists():
                print(f"✓ 相对路径验证通过: {expected_path} 存在")
            else:
                print(f"! 警告: 相对路径验证失败: {expected_path} 不存在")
                print(f"  如果您在不同的目录中运行train.py，请使用绝对路径或在同一目录中运行")
        else:
            print(f"! 错误: 未能创建必要的文件: {actual_path.absolute()}")
            print(f"  请检查文件权限和目录结构")
        
        # 检查是否在根目录也创建了dataset/SV2TTS结构
        root_sv2tts = Path('dataset/SV2TTS')
        if root_sv2tts.exists():
            print(f"✓ 在当前目录中找到dataset/SV2TTS目录: {root_sv2tts.absolute()}")
            if (root_sv2tts / "encoder" / "test" / "_sources.txt").exists():
                print(f"✓ 找到必要的_sources.txt文件在相对路径: dataset/SV2TTS/encoder/test/_sources.txt")
            else:
                # 尝试创建必要的文件结构
                try:
                    (root_sv2tts / "encoder").mkdir(exist_ok=True)
                    (root_sv2tts / "encoder" / "train").mkdir(exist_ok=True)
                    (root_sv2tts / "encoder" / "test").mkdir(exist_ok=True)
                    
                    with open(root_sv2tts / "encoder" / "_sources.txt", "w") as f:
                        f.write("")
                    with open(root_sv2tts / "encoder" / "train" / "_sources.txt", "w") as f:
                        f.write("")
                    with open(root_sv2tts / "encoder" / "test" / "_sources.txt", "w") as f:
                        f.write("")
                    
                    print(f"✓ 已在当前目录创建必要的文件结构")
                except Exception as e:
                    print(f"! 无法创建必要的文件结构: {e}")
        else:
            print(f"! 警告: 未在当前目录找到dataset/SV2TTS目录")
            print(f"  考虑在当前目录创建此结构或使用完整路径")
            # 尝试创建此结构
            try:
                root_sv2tts.mkdir(parents=True, exist_ok=True)
                (root_sv2tts / "encoder").mkdir(exist_ok=True)
                (root_sv2tts / "encoder" / "train").mkdir(exist_ok=True)
                (root_sv2tts / "encoder" / "test").mkdir(exist_ok=True)
                
                with open(root_sv2tts / "encoder" / "_sources.txt", "w") as f:
                    f.write("")
                with open(root_sv2tts / "encoder" / "train" / "_sources.txt", "w") as f:
                    f.write("")
                with open(root_sv2tts / "encoder" / "test" / "_sources.txt", "w") as f:
                    f.write("")
                
                print(f"✓ 已在当前目录创建dataset/SV2TTS结构")
            except Exception as e:
                print(f"! 无法创建结构: {e}")
    else:
        print("\n==========================")
        print("处理完成! 使用以下命令训练模型:")
        print(f"python ResNet/train.py <训练ID> {args.output_root} --resnet_type resnet50")
        print("==========================")
        
    return output_dir


if __name__ == '__main__':
    main()

