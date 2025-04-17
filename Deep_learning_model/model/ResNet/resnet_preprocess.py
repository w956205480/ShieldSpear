from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from pathlib import Path
from encoder import inference as encoder
from ResNet.inference import load_model as resnet_load_model
from ResNet.inference import is_loaded as resnet_is_loaded
from ResNet.inference import embed_utterance as resnet_embed_utterance
from tqdm import tqdm
import numpy as np


def embed_utterance_resnet(fpaths, resnet_model_fpath):
    """
    使用ResNet模型为一段语音生成说话人嵌入向量
    
    参数:
        fpaths: 包含wav文件路径和嵌入向量保存路径的元组
        resnet_model_fpath: ResNet模型路径
    """
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


def create_embeddings_resnet(synthesizer_root: Path, resnet_model_fpath: Path, n_processes: int):
    """
    使用ResNet模型为合成器训练数据创建说话人嵌入向量
    
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
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]
    
    # 在多个进程中并行处理
    func = partial(embed_utterance_resnet, resnet_model_fpath=resnet_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "使用ResNet生成嵌入向量", len(fpaths), unit="utterances"))
    
    print(f"已完成 {len(fpaths)} 个语音的ResNet嵌入向量生成")


if __name__ == "__main__":
    import argparse
    from utils.argutils import print_args
    
    parser = argparse.ArgumentParser(
        description="使用ResNet模型为合成器创建说话人嵌入向量",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "包含音频和train.txt文件的合成器训练数据路径")
    parser.add_argument("-r", "--resnet_model_fpath", type=Path,
                        default="ResNet/saved_models/resnet_encoder.pt", help=\
        "ResNet编码器模型路径(可使用标准ResNet预训练模型或自定义训练模型)")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help=\
        "并行处理进程数。如果CUDA内存不足，请降低此值")
    
    args = parser.parse_args()
    
    # 打印参数
    print_args(args, parser)
    
    # 执行预处理
    create_embeddings_resnet(
        args.synthesizer_root,
        args.resnet_model_fpath,
        args.n_processes
    ) 