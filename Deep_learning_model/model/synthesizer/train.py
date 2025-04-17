from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from time import perf_counter as timer
import numpy as np
from functools import partial
import time
import traceback
import os
import math

from synthesizer import audio
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils.__init__ import ValueWindow, data_parallel_workaround
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from vocoder.display import *


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def train(run_id: str, syn_dir: Path, models_dir: Path, save_every: int,  backup_every: int, force_restart: bool,
          hparams):
    # 首先运行元数据清理，确保所有文件都存在
    print("预处理：验证训练数据...")
    from pathlib import Path
    import sys
    import os

    # 添加项目根目录到路径
    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys_path not in sys.path:
        sys.path.append(sys_path)

    try:
        # 尝试导入清理脚本
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import clean_synthesizer_metadata
        print("自动运行元数据清理...")
        clean_synthesizer_metadata.clean_metadata(syn_dir, dry_run=False)
    except ImportError:
        print("清理脚本导入失败，将手动执行清理流程")
        # 手动执行清理逻辑
        synthesizer_root = Path(syn_dir)
        metadata_path = synthesizer_root.joinpath("train.txt")
        mel_dir = synthesizer_root.joinpath("mels")
        embed_dir = synthesizer_root.joinpath("embeds")
        
        # 读取元数据
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = [line.strip().split("|") for line in f if line.strip()]
        
        # 验证文件存在性
        from tqdm import tqdm
        valid_metadata = []
        
        for meta in tqdm(metadata, desc="验证文件"):
            if len(meta) < 3:
                continue
                
            mel_path = mel_dir.joinpath(meta[1])
            embed_path = embed_dir.joinpath(meta[2])
            
            if mel_path.exists() and embed_path.exists():
                valid_metadata.append(meta)
        
        # 如果有无效文件，更新元数据
        if len(valid_metadata) < len(metadata):
            print(f"发现 {len(metadata) - len(valid_metadata)} 个无效条目，更新元数据...")
            import shutil
            backup_path = metadata_path.with_suffix(".bak_train")
            shutil.copy(metadata_path, backup_path)
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                for meta in valid_metadata:
                    f.write("|".join(meta) + "\n")
            
            print(f"已更新元数据，保留 {len(valid_metadata)} 个有效条目")
            
    # 继续原来的训练流程
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)

    weights_fpath = model_dir / f"synthesizer.pt"
    metadata_fpath = syn_dir.joinpath("train.txt")

    print("Checkpoint path: {}".format(weights_fpath))
    print("Loading training data from: {}".format(metadata_fpath))
    print("Using model: Tacotron")

    # Bookkeeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    # From WaveRNN/train_tacotron.py
    if torch.cuda.is_available():
        device = torch.device("cuda")

        for session in hparams.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError("`batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Instantiate Tacotron Model
    print("\nInitialising Tacotron Model...\n")
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=hparams.tts_dropout,
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())

    # Load the weights
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of Tacotron from scratch\n")
        model.save(weights_fpath)

        # Embeddings metadata
        char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
        with open(char_embedding_fpath, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s

                f.write("{}\n".format(symbol))

    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("Tacotron weights loaded from step %d" % model.step)

    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.txt")
    mel_dir = syn_dir.joinpath("mels")
    embed_dir = syn_dir.joinpath("embeds")
    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)

    for i, session in enumerate(hparams.tts_schedule):
        current_step = model.get_step()

        r, lr, max_step, batch_size = session

        training_steps = max_step - current_step

        # Do we need to change to the next session?
        if current_step >= max_step:
            # Are there no further sessions than the current one?
            if i == len(hparams.tts_schedule) - 1:
                # We have completed training. Save the model and exit
                model.save(weights_fpath, optimizer)
                break
            else:
                # There is a following session, go to it
                continue

        model.r = r

        # Begin the training
        simple_table([(f"Steps with r={r}", str(training_steps // 1000) + "k Steps"),
                      ("Batch Size", batch_size),
                      ("Learning Rate", lr),
                      ("Outputs/Step (r)", model.r)])

        for p in optimizer.param_groups:
            p["lr"] = lr

        collate_fn = partial(collate_synthesizer, r=r, hparams=hparams)
        data_loader = DataLoader(
            dataset, 
            batch_size, 
            shuffle=True, 
            num_workers=2, 
            collate_fn=collate_fn, 
            drop_last=True,
            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )

        total_iters = len(dataset)
        steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
        epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)

        for epoch in range(1, epochs+1):
            print(f"\n开始第 {epoch}/{epochs} 轮训练")
            for i, batch_data in enumerate(data_loader, 1):
                try:
                    # 定期报告进度，即使在处理批次前
                    if i % 10 == 0 or i == 1:
                        print(f"处理批次 {i}/{steps_per_epoch}...")
                    
                    # 安全获取批次数据
                    if len(batch_data) != 4:
                        print(f"警告: 批次数据格式异常，跳过。长度: {len(batch_data)}")
                        continue
                        
                    texts, mels, embeds, idx = batch_data
                    
                    # 检查批次是否为空
                    if len(texts) == 0 or len(mels) == 0 or len(embeds) == 0:
                        print("警告: 空批次，跳过")
                        continue
                        
                    # 检查批次数据形状
                    print(f"批次数据形状: texts={texts.shape}, mels={mels.shape}, embeds={embeds.shape}")
                        
                    start_time = time.time()

                    # Generate stop tokens for training
                    stop = torch.ones(mels.shape[0], mels.shape[2])
                    for j, k in enumerate(idx):
                        stop[j, :int(dataset.metadata[k][4])-1] = 0

                    texts = texts.to(device)
                    mels = mels.to(device)
                    embeds = embeds.to(device)
                    stop = stop.to(device)

                    print("数据已移至设备，开始前向传播...")
                    
                    # Forward pass
                    # Parallelize model onto GPUS using workaround due to python bug
                    if device.type == "cuda" and torch.cuda.device_count() > 1:
                        m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(model, texts, mels, embeds)
                    else:
                        m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)

                    print("前向传播完成，计算损失...")
                    
                    # Backward pass
                    m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                    m2_loss = F.mse_loss(m2_hat, mels)
                    stop_loss = F.binary_cross_entropy(stop_pred, stop)

                    loss = m1_loss + m2_loss + stop_loss

                    print("计算梯度...")
                    optimizer.zero_grad()
                    loss.backward()

                    if hparams.tts_clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
                        if np.isnan(grad_norm.cpu()):
                            print("grad_norm was NaN!")

                    print("更新参数...")
                    optimizer.step()

                    time_window.append(time.time() - start_time)
                    loss_window.append(loss.item())

                    step = model.get_step()
                    k = step // 1000

                    msg = f"| Epoch: {epoch}/{epochs} ({i}/{steps_per_epoch}) | Loss: {loss_window.average:#.4} | " \
                          f"{1./time_window.average:#.2} steps/s | Step: {k}k | "
                    stream(msg)

                    # 每20批次保存一次模型，确保不会丢失太多进度
                    if i % 20 == 0:
                        print(f"\n暂存模型 step={step}...")
                        model.save(weights_fpath, optimizer)
                        print("暂存完成，继续训练...")

                    # Backup or save model as appropriate
                    if backup_every != 0 and step % backup_every == 0 :
                        backup_fpath = weights_fpath.parent / f"synthesizer_{k:06d}.pt"
                        model.save(backup_fpath, optimizer)

                    if save_every != 0 and step % save_every == 0 :
                        # Must save latest optimizer state to ensure that resuming training
                        # doesn't produce artifacts
                        model.save(weights_fpath, optimizer)

                    # Evaluate model to generate samples
                    epoch_eval = hparams.tts_eval_interval == -1 and i == steps_per_epoch  # If epoch is done
                    step_eval = hparams.tts_eval_interval > 0 and step % hparams.tts_eval_interval == 0  # Every N steps
                    if epoch_eval or step_eval:
                        for sample_idx in range(hparams.tts_eval_num_samples):
                            # At most, generate samples equal to number in the batch
                            if sample_idx + 1 <= len(texts):
                                # Remove padding from mels using frame length in metadata
                                mel_length = int(dataset.metadata[idx[sample_idx]][4])
                                mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                                target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                                attention_len = mel_length // model.r

                                eval_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                                           mel_prediction=mel_prediction,
                                           target_spectrogram=target_spectrogram,
                                           input_seq=np_now(texts[sample_idx]),
                                           step=step,
                                           plot_dir=plot_dir,
                                           mel_output_dir=mel_output_dir,
                                           wav_dir=wav_dir,
                                           sample_num=sample_idx + 1,
                                           loss=loss,
                                           hparams=hparams)

                    # Break out of loop to update training schedule
                    if step >= max_step:
                        print(f"已达到最大步数 {max_step}，结束当前训练阶段。")
                        break

                except Exception as e:
                    print(f"异常: {e}")
                    print(f"详细错误: {traceback.format_exc()}")
                    print("跳过当前批次，继续训练...")
                    continue

            # Add line break after every epoch
            print("")


def eval_model(attention, mel_prediction, target_spectrogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath("attention_step_{}_sample_{}".format(step, sample_num)))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))
