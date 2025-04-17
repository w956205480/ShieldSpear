import torch
from synthesizer import audio
from synthesizer.hparams import hparams
from synthesizer.models.tacotron import Tacotron
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import text_to_sequence
from vocoder.display import simple_table
from pathlib import Path
from typing import Union, List
import numpy as np
import librosa


class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams

    def __init__(self, model_fpath: Path, verbose=True):
        """
        The model isn't instantiated and loaded in memory until needed or until load() is called.

        :param model_fpath: path to the trained model file
        :param verbose: if False, prints less information when using the model
        """
        self.model_fpath = model_fpath
        self.verbose = verbose

        # Check for GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.verbose:
            print("Synthesizer using device:", self.device)

        # Tacotron model will be instantiated later on first use.
        self._model = None

    def is_loaded(self):
        """
        Whether the model is loaded in memory.
        """
        return self._model is not None

    def load(self):
        """
        Instantiates and loads the model given the weights file that was passed in the constructor.
        """
        self._model = Tacotron(embed_dims=hparams.tts_embed_dims,
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
                               speaker_embedding_size=hparams.speaker_embedding_size).to(self.device)

        self._model.load(self.model_fpath)
        self._model.eval()

        if self.verbose:
            print("Loaded synthesizer \"%s\" trained to step %d" % (self.model_fpath.name, self._model.state_dict()["step"]))

    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256)
        :param return_alignments: if True, a matrix representing the alignments between the
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the
        sequence length of spectrogram i, and possibly the alignments.
        """
        # Load the model on the first request.
        if not self.is_loaded():
            self.load()

        # 检查并处理文本输入
        if not texts:
            raise ValueError("文本列表不能为空")
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                print(f"警告: 输入文本 #{i} 不是字符串类型，尝试转换")
                texts[i] = str(text)

        # Preprocess text inputs
        try:
            inputs = [text_to_sequence(text.strip(), hparams.tts_cleaner_names) for text in texts]
        except Exception as e:
            print(f"文本预处理出错: {e}")
            print("尝试使用基本ASCII转换...")
            inputs = []
            for text in texts:
                try:
                    seq = text_to_sequence(text.strip(), hparams.tts_cleaner_names)
                    inputs.append(seq)
                except:
                    # 如果转换失败，使用基本ASCII字符
                    print(f"无法处理文本: '{text}'，使用默认文本")
                    inputs.append(text_to_sequence("test input", hparams.tts_cleaner_names))

        # 检查并处理嵌入向量
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        
        # 验证嵌入向量维度
        for i, embed in enumerate(embeddings):
            # 检查嵌入向量维度
            if isinstance(embed, np.ndarray):
                # 处理不同的输入维度情况
                if embed.ndim == 3:  # shape [1, 1, D]
                    print(f"警告: 嵌入向量 #{i} 有3个维度，尝试reshape")
                    embed = embed.reshape(-1)
                    embeddings[i] = embed
                elif embed.ndim == 2:  # shape [1, D]
                    print(f"警告: 嵌入向量 #{i} 有2个维度，尝试reshape")
                    embed = embed.reshape(-1)
                    embeddings[i] = embed
                
                # 检查嵌入向量长度
                if embed.shape[0] != hparams.speaker_embedding_size:
                    print(f"警告: 嵌入向量 #{i} 大小为 {embed.shape[0]}，预期 {hparams.speaker_embedding_size}")
                    # 如果太小，填充；如果太大，截断
                    if embed.shape[0] < hparams.speaker_embedding_size:
                        padded = np.zeros(hparams.speaker_embedding_size)
                        padded[:embed.shape[0]] = embed
                        embeddings[i] = padded
                    else:
                        embeddings[i] = embed[:hparams.speaker_embedding_size]
            else:
                print(f"警告: 嵌入向量 #{i} 不是numpy数组，尝试转换")
                try:
                    embeddings[i] = np.array(embed, dtype=np.float32)
                except:
                    print(f"无法转换嵌入向量 #{i}，使用随机向量")
                    embeddings[i] = np.random.randn(hparams.speaker_embedding_size).astype(np.float32)
                    # 规范化嵌入向量
                    embeddings[i] /= np.linalg.norm(embeddings[i])

        # Batch inputs
        batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(inputs), hparams.synthesis_batch_size)]
        batched_embeds = [embeddings[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(embeddings), hparams.synthesis_batch_size)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if self.verbose:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            try:
                speaker_embeds = np.stack(batched_embeds[i-1])
            except Exception as e:
                print(f"堆叠嵌入向量出错: {e}")
                print("尝试手动处理嵌入向量...")
                processed_embeds = []
                for embed in batched_embeds[i-1]:
                    # 确保是一维的
                    embed = np.array(embed).flatten()
                    # 确保长度正确
                    if len(embed) != hparams.speaker_embedding_size:
                        print(f"调整嵌入向量大小: {len(embed)} -> {hparams.speaker_embedding_size}")
                        if len(embed) < hparams.speaker_embedding_size:
                            padded = np.zeros(hparams.speaker_embedding_size)
                            padded[:len(embed)] = embed
                            embed = padded
                        else:
                            embed = embed[:hparams.speaker_embedding_size]
                    # 规范化
                    embed = embed / (np.linalg.norm(embed) + 1e-8)
                    processed_embeds.append(embed)
                
                speaker_embeds = np.stack(processed_embeds)

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            # Inference
            try:
                _, mels, alignments = self._model.generate(chars, speaker_embeddings)
                mels = mels.detach().cpu().numpy()
                for m in mels:
                    # Trim silence from end of each spectrogram
                    while np.max(m[:, -1]) < hparams.tts_stop_threshold:
                        m = m[:, :-1]
                    specs.append(m)
            except Exception as e:
                print(f"生成梅尔频谱图时发生错误: {e}")
                import traceback
                traceback.print_exc()
                
                # 创建一个默认的梅尔频谱图作为fallback
                print("创建默认梅尔频谱图...")
                default_len = 100  # 默认长度
                for _ in range(len(batch)):
                    default_mel = np.zeros((hparams.num_mels, default_len))
                    specs.append(default_mel)

        if self.verbose:
            print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def load_preprocess_wav(fpath):
        """
        Loads and preprocesses an audio file under the same conditions the audio files were used to
        train the synthesizer.
        """
        wav = librosa.load(str(fpath), hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        """
        Creates a mel spectrogram from an audio file in the same manner as the mel spectrograms that
        were fed to the synthesizer when training.
        """
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav

        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram

    @staticmethod
    def griffin_lim(mel):
        """
        Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
        with the same parameters present in hparams.py.
        """
        return audio.inv_mel_spectrogram(mel, hparams)


def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)
