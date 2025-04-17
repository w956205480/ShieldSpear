from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
from pathlib import Path

# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        
    def _load_utterances(self):
        # 首先尝试在说话人目录中查找_sources.txt
        speaker_sources_path = self.root.joinpath("_sources.txt")
        
        if speaker_sources_path.exists():
            # 如果说话人目录中存在_sources.txt文件，使用传统方式读取
            with speaker_sources_path.open("r") as sources_file:
                sources = [l.split(",") for l in sources_file]
            sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
            self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
        else:
            # 如果说话人目录中不存在_sources.txt，尝试在父目录（训练或测试目录）中查找
            parent_sources_path = self.root.parent.joinpath("_sources.txt")
            
            if not parent_sources_path.exists():
                # 如果父目录也没有_sources.txt，尝试在更上层目录查找
                # 可能的SV2TTS结构：.../encoder/train/speaker 或 .../encoder/test/speaker
                encoder_sources_path = self.root.parent.parent.joinpath("_sources.txt")
                if encoder_sources_path.exists():
                    parent_sources_path = encoder_sources_path
            
            if parent_sources_path.exists():
                # 从主_sources.txt中过滤出当前说话人的条目
                speaker_prefix = f"{self.name}/"
                sources = []
                
                with parent_sources_path.open("r") as sources_file:
                    for line in sources_file:
                        # 检查该行是否属于当前说话人
                        parts = line.strip().split(",")
                        if len(parts) >= 2 and parts[0].startswith(speaker_prefix):
                            # 移除说话人前缀，只保留文件名部分
                            frames_fname = parts[0][len(speaker_prefix):]
                            wave_fpath = parts[1]
                            sources.append((frames_fname, wave_fpath))
                
                # 创建话语对象
                self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources]
            else:
                # 如果所有可能的位置都没有找到_sources.txt，尝试直接查找npy文件
                print(f"警告：找不到说话人 {self.name} 的_sources.txt文件，尝试直接搜索音频文件")
                npy_files = list(self.root.glob("*.npy"))
                if not npy_files:
                    raise FileNotFoundError(f"无法找到说话人 {self.name} 的任何语音文件或_sources.txt")
                
                # 使用npy文件路径作为源
                self.utterances = [Utterance(f, str(f.with_suffix(".wav"))) for f in npy_files]
        
        # 创建循环器
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all 
        utterances come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial utterances to sample from the set of utterances from 
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than 
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance, 
        frames are the frames of the partial utterances and range is the range of the partial 
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a
