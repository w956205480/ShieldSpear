from utils.argutils import print_args
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import Pool
from encoder import inference as encoder
from ResNet.inference import load_model as resnet_load_model
from ResNet.inference import is_loaded as resnet_is_loaded
from ResNet.inference import embed_utterance as resnet_embed_utterance
import shutil


def embed_utterance_safe(fpaths, resnet_model_fpath):
    """使用ResNet模型为一段语音生成说话人嵌入向量，增加错误处理
    
    参数:
        fpaths: 包含wav文件路径和嵌入向量保存路径的元组
        resnet_model_fpath: ResNet模型路径
    返回:
        bool: 处理是否成功
    """
    try:
        # 确保模型已加载
        if not resnet_is_loaded():
            resnet_load_model(resnet_model_fpath)
        
        # 计算说话人嵌入向量
        wav_fpath, embed_fpath = fpaths
        wav = np.load(wav_fpath)
        wav = encoder.preprocess_wav(wav)  # 使用相同的音频预处理
        embed = resnet_embed_utterance(wav)
        
        # 保存嵌入向量
        np.save(embed_fpath, embed, allow_pickle=False)
        return True
    except Exception as e:
        print(f"处理文件时出错: {fpaths}")
        print(f"错误信息: {str(e)}")
        return False


def create_embeddings_safe(synthesizer_root, resnet_model_fpath, n_processes):
    """使用ResNet模型为合成器训练数据创建说话人嵌入向量，跳过不存在的文件
    
    参数:
        synthesizer_root: 合成器数据根目录
        resnet_model_fpath: ResNet模型路径
        n_processes: 并行进程数
    """
    # 检查必要的目录和文件
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists(), "音频目录或元数据文件不存在！"
    
    # 创建嵌入向量保存目录
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)
    
    # 收集输入波形文件路径和目标输出嵌入向量文件路径
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    
    # 验证文件存在性并筛选
    valid_metadata = []
    valid_fpaths = []
    
    print("检查文件存在性...")
    for meta in tqdm(metadata, desc="验证文件"):
        wav_path = wav_dir.joinpath(meta[0])
        if wav_path.exists() and wav_path.stat().st_size > 0:
            valid_metadata.append(meta)
            valid_fpaths.append((wav_path, embed_dir.joinpath(meta[2])))
        else:
            print(f"跳过不存在或空文件: {wav_path}")
    
    print(f"原始数据条目: {len(metadata)}, 有效数据条目: {len(valid_metadata)}")
    
    # 如果有无效文件，备份并重新生成元数据
    if len(valid_metadata) < len(metadata):
        backup_path = metadata_fpath.with_suffix(".bak")
        shutil.copy(metadata_fpath, backup_path)
        print(f"原始元数据已备份至: {backup_path}")
        
        # 写入新的元数据文件
        with metadata_fpath.open("w") as f:
            for meta in valid_metadata:
                f.write("|".join(meta) + "\n")
        
        print(f"新元数据文件已写入，包含 {len(valid_metadata)} 个条目")
    
    # 仅在有有效文件时生成嵌入向量
    if len(valid_fpaths) > 0:
        # 在多个进程中并行处理
        func = partial(embed_utterance_safe, resnet_model_fpath=resnet_model_fpath)
        
        if n_processes > 1:
            # 多进程处理
            job = Pool(n_processes).imap(func, valid_fpaths)
            results = list(tqdm(job, "使用ResNet生成嵌入向量", len(valid_fpaths), unit="utterances"))
        else:
            # 单进程处理，便于查看错误
            results = []
            for fpath_pair in tqdm(valid_fpaths, "使用ResNet生成嵌入向量"):
                result = func(fpath_pair)
                results.append(result)
        
        success_count = results.count(True)
        failed_count = results.count(False)
        
        print(f"处理完成: 成功 {success_count} 个，失败 {failed_count} 个")
        
        # 如果有失败的处理，再次更新元数据文件
        if failed_count > 0:
            print("检测到处理失败的项目，更新元数据文件...")
            
            # 创建新的元数据文件
            final_metadata = []
            for i, (result, meta) in enumerate(zip(results, valid_metadata)):
                if result:
                    final_metadata.append(meta)
            
            # 再次备份
            backup_path = metadata_fpath.with_suffix(".bak2")
            shutil.copy(metadata_fpath, backup_path)
            
            # 写入最终的元数据文件
            with metadata_fpath.open("w") as f:
                for meta in final_metadata:
                    f.write("|".join(meta) + "\n")
            
            print(f"最终元数据文件已更新，包含 {len(final_metadata)} 个条目")
        
        return success_count, failed_count
    else:
        print("没有找到有效文件，嵌入向量生成已跳过")
        return 0, len(metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用ResNet模型为合成器创建说话人嵌入向量",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "包含音频和train.txt文件的合成器训练数据路径。"
        "如果使用默认设置，应该是<datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-r", "--resnet_model_fpath", type=Path,
                        default="ResNet/saved_models/resnet_encoder.pt", help=\
        "ResNet编码器模型路径(可使用标准ResNet预训练模型或通过encoder_train_resnet.py训练的模型)")
    parser.add_argument("-n", "--n_processes", type=int, default=1, help=\
        "并行处理进程数。每个进程会创建一个编码器，因此在低内存GPU上可能需要降低此值。"
        "如果CUDA出现问题，请设置为1")
    parser.add_argument("--safe", action="store_true", help=\
        "启用安全模式，使用单进程处理并显示详细错误信息")
    args = parser.parse_args()

    # 如果启用安全模式，使用单线程
    if args.safe:
        args.n_processes = 1
        print("已启用安全模式，使用单线程处理")

    # 打印参数
    print_args(args, parser)
    
    # 使用ResNet处理
    print("使用ResNet模型生成嵌入向量，增强版（自动处理缺失文件）...")
    success, failed = create_embeddings_safe(
        args.synthesizer_root,
        args.resnet_model_fpath,
        args.n_processes
    )
    
    print(f"处理完成! 最终结果: 成功 {success} 个，跳过/失败 {failed} 个") 