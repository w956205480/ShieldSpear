import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union


class HighwayNetwork(nn.Module):
    """Highway网络，允许信息直接通过或者通过变换后传递
    在深层网络中帮助梯度流动，类似于残差连接但带有门控机制"""
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)  # 变换网络
        self.W2 = nn.Linear(size, size)  # 门控网络
        self.W1.bias.data.fill_(0.)  # 初始化变换网络的偏置为0

    def forward(self, x):
        x1 = self.W1(x)  # 变换输入
        x2 = self.W2(x)  # 计算门控值
        g = torch.sigmoid(x2)  # 门控信号，值域为(0,1)
        y = g * F.relu(x1) + (1. - g) * x  # 门控机制：g控制信息流过变换网络的比例，(1-g)控制直接通过的比例
        return y


class Encoder(nn.Module):
    """文本编码器，将文本转换为隐藏表示，使用CBHG架构"""
    def __init__(self, embed_dims, num_chars, encoder_dims, K, num_highways, dropout):
        super().__init__()
        prenet_dims = (encoder_dims, encoder_dims)  # PreNet层维度
        cbhg_channels = encoder_dims  # CBHG通道数
        self.embedding = nn.Embedding(num_chars, embed_dims)  # 字符嵌入层
        self.pre_net = PreNet(embed_dims, fc1_dims=prenet_dims[0], fc2_dims=prenet_dims[1],
                              dropout=dropout)  # PreNet网络用于处理嵌入向量
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                         proj_channels=[cbhg_channels, cbhg_channels],
                         num_highways=num_highways)  # CBHG模块是Encoder的核心

    def forward(self, x, speaker_embedding=None):
        x = self.embedding(x)  # 将输入文本转换为嵌入向量
        x = self.pre_net(x)  # 通过PreNet网络处理嵌入向量
        x.transpose_(1, 2)  # 转置张量为(batch_size, channels, time_steps)格式
        x = self.cbhg(x)  # 通过CBHG模块处理
        if speaker_embedding is not None:
            x = self.add_speaker_embedding(x, speaker_embedding)  # 如果提供了说话人嵌入，则添加到编码器输出
        return x

    def add_speaker_embedding(self, x, speaker_embedding):
        """将说话人嵌入与编码器输出连接
        SV2TTS：将说话人嵌入重复并与每个字符的编码输出连接"""
        # 输入x是编码器输出，形状为(batch_size, num_chars, tts_embed_dims)
        # 训练时，speaker_embedding是形状为(batch_size, speaker_embedding_size)的2D张量
        # 推理时，speaker_embedding是形状为(speaker_embedding_size)的1D张量

        # 保存便于理解的维度命名
        batch_size = x.size()[0]
        num_chars = x.size()[1]

        if speaker_embedding.dim() == 1:
            idx = 0  # 如果是1D张量（推理时），索引为0
        else:
            idx = 1  # 如果是2D张量（训练时），索引为1

        # 复制说话人嵌入以匹配输入文本长度
        speaker_embedding_size = speaker_embedding.size()[idx]
        e = speaker_embedding.repeat_interleave(num_chars, dim=idx)  # 重复说话人嵌入

        # 重塑并转置
        e = e.reshape(batch_size, speaker_embedding_size, num_chars)
        e = e.transpose(1, 2)  # 形状变为(batch_size, num_chars, speaker_embedding_size)

        # 将重复的说话人嵌入与编码器输出连接
        x = torch.cat((x, e), 2)  # 沿最后一个维度连接
        return x


class BatchNormConv(nn.Module):
    """具有批归一化的1D卷积层"""
    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)  # 批归一化层
        self.relu = relu  # 是否使用ReLU激活

    def forward(self, x):
        x = self.conv(x)  # 卷积操作
        x = F.relu(x) if self.relu is True else x  # 条件性应用ReLU
        return self.bnorm(x)  # 应用批归一化


class CBHG(nn.Module):
    """CBHG模块：1D卷积组 + Highway网络 + 双向GRU
    是Tacotron中的关键组件，用于编码器和后处理网络"""
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()

        # 用于调用`flatten_parameters()`的RNN列表
        self._to_flatten = []

        # 创建卷积组
        self.bank_kernels = [i for i in range(1, K + 1)]  # 卷积核大小从1到K
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)  # 为每个核大小创建一个卷积
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)  # 最大池化层

        # 投影卷积
        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        # 如果需要，修复Highway输入
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False

        # Highway网络
        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        # 双向GRU
        self.rnn = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)

        # 避免RNN参数碎片化和相关警告
        self._flatten_parameters()

    def forward(self, x):
        # 虽然在初始化时调用了`_flatten_parameters()`，但在使用DataParallel时
        # 模型会被复制，不再保证权重在GPU内存中是连续的，因此需要再次调用
        self._flatten_parameters()

        # 保存这些以便后续使用
        residual = x  # 残差连接
        seq_len = x.size(-1)  # 序列长度
        conv_bank = []  # 存储卷积组输出

        # 卷积组处理
        for conv in self.conv1d_bank:
            c = conv(x)  # 卷积操作
            conv_bank.append(c[:, :, :seq_len])  # 保持序列长度一致

        # 沿通道轴堆叠
        conv_bank = torch.cat(conv_bank, dim=1)

        # 最大池化并保持与残差相同的长度
        x = self.maxpool(conv_bank)[:, :, :seq_len]

        # 1D卷积投影
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # 残差连接
        x = x + residual

        # 通过Highway网络
        x = x.transpose(1, 2)  # 转置为(batch, time_steps, channels)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)  # 如有必要，调整维度
        for h in self.highways: x = h(x)  # 通过所有Highway层

        # 最后通过RNN
        x, _ = self.rnn(x)  # 双向GRU处理
        return x

    def _flatten_parameters(self):
        """对所有RNN调用`flatten_parameters`以提高效率并避免PyTorch警告"""
        [m.flatten_parameters() for m in self._to_flatten]

class PreNet(nn.Module):
    """PreNet：两个带dropout的全连接层
    在Tacotron中用于编码器和解码器，dropout在训练和推理时都启用"""
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)  # 第一个全连接层
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)  # 第二个全连接层
        self.p = dropout  # dropout概率

    def forward(self, x):
        x = self.fc1(x)  # 第一层变换
        x = F.relu(x)  # ReLU激活
        x = F.dropout(x, self.p, training=True)  # 注意：即使在推理时也应用dropout
        x = self.fc2(x)  # 第二层变换
        x = F.relu(x)  # ReLU激活
        x = F.dropout(x, self.p, training=True)  # 推理时也应用dropout，这是Tacotron的特性
        return x


class Attention(nn.Module):
    """基本注意力机制"""
    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)  # 查询变换
        self.v = nn.Linear(attn_dims, 1, bias=False)  # 能量函数

    def forward(self, encoder_seq_proj, query, t):
        # 变换查询向量
        query_proj = self.W(query).unsqueeze(1)  # 形状变为(batch_size, 1, attn_dims)

        # 计算注意力分数
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))  # 计算能量
        scores = F.softmax(u, dim=1)  # 计算注意力权重

        return scores.transpose(1, 2)  # 转置为(batch_size, 1, time_steps)


class LSA(nn.Module):
    """位置敏感注意力(Location Sensitive Attention)
    考虑先前注意力的累积，有助于单调对齐"""
    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=True)
        self.L = nn.Linear(filters, attn_dim, bias=False)  # 位置特征处理
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)  # 在此项中包含注意力偏置
        self.v = nn.Linear(attn_dim, 1, bias=False)  # 能量函数
        self.cumulative = None  # 累积注意力权重
        self.attention = None  # 当前注意力权重

    def init_attention(self, encoder_seq_proj):
        """初始化注意力权重和累积注意力"""
        try:
            device = next(self.parameters()).device  # 使用与参数相同的设备
        except StopIteration:
            # 如果模型没有参数，使用输入张量的设备
            print("警告：模型没有参数，使用输入张量的设备")
            device = encoder_seq_proj.device
            
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t, device=device)  # 初始化为零
        self.attention = torch.zeros(b, t, device=device)  # 初始化为零

    def forward(self, encoder_seq_proj, query, t, chars):
        """
        计算注意力权重
        encoder_seq_proj: 编码器输出的投影
        query: 解码器的当前隐藏状态
        t: 当前时间步
        chars: 字符序列(用于屏蔽填充)
        """
        if t == 0: self.init_attention(encoder_seq_proj)  # 如果是第一步，初始化注意力

        # 处理查询
        processed_query = self.W(query).unsqueeze(1)  # 形状(batch_size, 1, attn_dim)

        # 处理位置特征
        location = self.cumulative.unsqueeze(1)  # 形状(batch_size, 1, time_steps)
        processed_loc = self.L(self.conv(location).transpose(1, 2))  # 卷积和线性变换

        # 计算能量
        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)  # 去除最后一个维度

        # 屏蔽零填充字符
        u = u * (chars != 0).float()  # 应用掩码，使填充位置的分数为0

        # 计算平滑的注意力权重
        scores = F.softmax(u, dim=1)  # softmax归一化
        self.attention = scores  # 保存当前注意力
        self.cumulative = self.cumulative + self.attention  # 更新累积注意力

        return scores.unsqueeze(-1).transpose(1, 2)  # 返回形状(batch_size, 1, time_steps)


class Decoder(nn.Module):
    """Tacotron解码器
    使用注意力机制从编码器输出生成梅尔频谱图"""
    # 类变量，因为其值在类之间不变
    # 但应由类限定作用域，因为它是Decoder的属性
    max_r = 20  # 最大reduction factor
    def __init__(self, n_mels, encoder_dims, decoder_dims, lstm_dims,
                 dropout, speaker_embedding_size):
        super().__init__()
        self.register_buffer("r", torch.tensor(1, dtype=torch.int))  # reduction factor，控制每步生成的帧数
        self.n_mels = n_mels  # 梅尔频谱图的维度
        prenet_dims = (decoder_dims * 2, decoder_dims * 2)  # PreNet层尺寸
        self.prenet = PreNet(n_mels, fc1_dims=prenet_dims[0], fc2_dims=prenet_dims[1],
                             dropout=dropout)  # 解码器的PreNet
        self.attn_net = LSA(decoder_dims)  # 位置敏感注意力
        # 注意力RNN，结合编码器输出和PreNet输出
        self.attn_rnn = nn.GRUCell(encoder_dims + prenet_dims[1] + speaker_embedding_size, decoder_dims)
        # RNN输入处理，连接编码器输出、解码器状态和说话人嵌入
        self.rnn_input = nn.Linear(encoder_dims + decoder_dims + speaker_embedding_size, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)  # 第一个残差LSTM
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)  # 第二个残差LSTM
        # 梅尔谱生成，输出可包含多个帧(最多max_r个)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)
        # 停止标记预测
        self.stop_proj = nn.Linear(encoder_dims + speaker_embedding_size + lstm_dims, 1)

    def zoneout(self, prev, current, p=0.1):
        """Zoneout：随机让一些单元保持前一状态而不更新
        有助于防止过拟合，类似于dropout但用于RNN"""
        try:
            device = next(self.parameters()).device  # 使用与参数相同的设备
        except StopIteration:
            # 如果模型没有参数，使用输入张量的设备
            device = prev.device
            
        mask = torch.zeros(prev.size(), device=device).bernoulli_(p)  # 创建随机二进制掩码
        return prev * mask + current * (1 - mask)  # 加权组合前一状态和当前状态

    def forward(self, encoder_seq, encoder_seq_proj, prenet_in,
                hidden_states, cell_states, context_vec, t, chars):
        """
        解码器前向传播
        encoder_seq: 编码器输出序列
        encoder_seq_proj: 编码器输出的投影
        prenet_in: 前一步生成的梅尔帧或初始帧
        hidden_states: 隐藏状态元组(attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states: 单元状态元组(rnn1_cell, rnn2_cell)
        context_vec: 上下文向量
        t: 当前时间步
        chars: 字符序列(用于注意力掩码)
        """
        # 需要这个来重塑梅尔输出
        batch_size = encoder_seq.size(0)

        # 解包隐藏状态和单元状态
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # 注意力RNN的PreNet处理
        prenet_out = self.prenet(prenet_in)

        # 计算注意力RNN的隐藏状态
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)  # 连接上下文向量和PreNet输出
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)  # 更新注意力RNN隐藏状态

        # 计算注意力分数
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t, chars)

        # 点积创建上下文向量
        context_vec = scores @ encoder_seq  # 加权求和编码器输出
        context_vec = context_vec.squeeze(1)

        # 连接注意力RNN输出和上下文向量并投影
        x = torch.cat([context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)

        # 计算第一个残差RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)  # 训练时应用zoneout
        else:
            rnn1_hidden = rnn1_hidden_next  # 推理时直接使用新状态
        x = x + rnn1_hidden  # 残差连接

        # 计算第二个残差RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)  # 训练时应用zoneout
        else:
            rnn2_hidden = rnn2_hidden_next  # 推理时直接使用新状态
        x = x + rnn2_hidden  # 残差连接

        # 投影生成梅尔输出
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]  # 重塑并截取所需的帧数
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)  # 打包隐藏状态
        cell_states = (rnn1_cell, rnn2_cell)  # 打包单元状态

        # 停止标记预测
        s = torch.cat((x, context_vec), dim=1)  # 连接RNN状态和上下文向量
        s = self.stop_proj(s)  # 线性变换
        stop_tokens = torch.sigmoid(s)  # sigmoid激活

        return mels, scores, hidden_states, cell_states, context_vec, stop_tokens


class Tacotron(nn.Module):
    """Tacotron模型：序列到序列的文本到语音合成模型
    结合编码器、解码器和后处理网络"""
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, 
                 fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways,
                 dropout, stop_threshold, speaker_embedding_size):
        super().__init__()
        self.n_mels = n_mels  # 梅尔频谱图的维度
        self.lstm_dims = lstm_dims  # LSTM的隐藏层维度
        self.encoder_dims = encoder_dims  # 编码器的输出维度
        self.decoder_dims = decoder_dims  # 解码器的隐藏层维度
        self.speaker_embedding_size = speaker_embedding_size  # 说话人嵌入的维度
        # 编码器：将文本转换为隐藏表示
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims,
                               encoder_K, num_highways, dropout)
        # 编码器输出投影：调整维度以匹配解码器
        self.encoder_proj = nn.Linear(encoder_dims + speaker_embedding_size, decoder_dims, bias=False)
        # 解码器：生成梅尔频谱图
        self.decoder = Decoder(n_mels, encoder_dims, decoder_dims, lstm_dims,
                               dropout, speaker_embedding_size)
        # 后处理网络：将梅尔频谱图转换为线性频谱图
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims,
                            [postnet_dims, fft_bins], num_highways)
        # 后处理投影：最终线性变换
        self.post_proj = nn.Linear(postnet_dims, fft_bins, bias=False)

        self.init_model()  # 初始化模型参数
        self.num_params()  # 输出参数数量

        # 注册缓冲区用于跟踪训练步骤和停止阈值
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("stop_threshold", torch.tensor(stop_threshold, dtype=torch.float32))

    @property
    def r(self):
        """获取当前的reduction factor
        每步解码器输出的梅尔帧数"""
        return self.decoder.r.item()

    @r.setter
    def r(self, value):
        """设置reduction factor的值"""
        self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def forward(self, x, m, speaker_embedding):
        """模型前向传播
        x: 输入文本
        m: 目标梅尔频谱图
        speaker_embedding: 说话人嵌入向量
        """
        # 增加训练步骤计数
        self.step += 1
        
        # 安全获取设备
        try:
            device = next(self.parameters()).device  # 使用与参数相同的设备
        except StopIteration:
            # 如果模型没有参数，使用输入张量的设备
            print("警告：模型没有参数，使用输入张量的设备")
            device = x.device

        # 获取序列长度和批次大小
        batch_size, _, steps = m.size()

        # 初始化所有隐藏状态并打包为元组
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # 初始化所有lstm单元状态并打包为元组
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO>帧作为解码器循环的起始
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # 初始上下文向量
        context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)

        # SV2TTS: 使用说话人嵌入运行编码器
        # 投影避免在解码器循环中进行不必要的矩阵乘法
        try:
            encoder_seq = self.encoder(x, speaker_embedding)  # 编码输入文本
            encoder_seq_proj = self.encoder_proj(encoder_seq)  # 投影编码器输出
        except Exception as e:
            print(f"编码器处理时发生错误: {e}")
            # 创建一个合理的默认值
            encoder_seq = torch.zeros(batch_size, x.size(1), self.encoder_dims + self.speaker_embedding_size, device=device)
            encoder_seq_proj = torch.zeros(batch_size, x.size(1), self.decoder_dims, device=device)

        # 需要几个列表来存储输出
        mel_outputs, attn_scores, stop_outputs = [], [], []

        # 运行解码器循环
        for t in range(0, steps, self.r):
            try:
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame  # 选择前一个输出或<GO>帧
                mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                hidden_states, cell_states, context_vec, t, x)
                mel_outputs.append(mel_frames)  # 收集梅尔输出
                attn_scores.append(scores)  # 收集注意力分数
                stop_outputs.extend([stop_tokens] * self.r)  # 收集停止标记
            except Exception as e:
                print(f"解码器循环中发生错误: {e}")
                # 创建一些合理的默认值
                mel_frames = torch.zeros(batch_size, self.n_mels, self.r, device=device)
                scores = torch.zeros(batch_size, 1, x.size(1), device=device)
                stop_tokens = torch.zeros(batch_size, 1, device=device)
                
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                stop_outputs.extend([stop_tokens] * self.r)

        # 将梅尔输出连接成序列
        try:
            mel_outputs = torch.cat(mel_outputs, dim=2)
        except Exception as e:
            print(f"连接梅尔输出时发生错误: {e}")
            mel_outputs = torch.zeros(batch_size, self.n_mels, steps, device=device)

        # 后处理生成线性频谱图
        try:
            postnet_out = self.postnet(mel_outputs)  # 通过CBHG后处理网络
            linear = self.post_proj(postnet_out)  # 最终投影
            linear = linear.transpose(1, 2)  # 转置为(batch, time, dim)格式
        except Exception as e:
            print(f"后处理网络中发生错误: {e}")
            linear = torch.zeros(batch_size, steps, self.n_mels, device=device)

        # 方便可视化
        try:
            attn_scores = torch.cat(attn_scores, 1)  # 连接所有注意力分数
            stop_outputs = torch.cat(stop_outputs, 1)  # 连接所有停止标记
        except Exception as e:
            print(f"连接注意力分数或停止标记时发生错误: {e}")
            attn_scores = torch.zeros(batch_size, steps // self.r, x.size(1), device=device)
            stop_outputs = torch.zeros(batch_size, steps, device=device)

        return mel_outputs, linear, attn_scores, stop_outputs

    def generate(self, x, speaker_embedding=None, steps=2000):
        """生成模式(推理)
        x: 输入文本
        speaker_embedding: 说话人嵌入向量
        steps: 最大生成步数
        """
        self.eval()  # 设置为评估模式
        
        # 安全获取设备
        try:
            device = next(self.parameters()).device  # 使用与参数相同的设备
        except StopIteration:
            # 如果模型没有参数，使用输入张量的设备
            device = x.device
            print("警告：模型没有参数，使用输入张量的设备")

        batch_size, _  = x.size()  # 获取批次大小

        # 初始化所有隐藏状态并打包为元组
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # 初始化所有lstm单元状态并打包为元组
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # 需要一个<GO>帧作为解码器循环的起始
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # 初始上下文向量
        context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)

        # SV2TTS: 使用说话人嵌入运行编码器
        # 投影避免在解码器循环中进行不必要的矩阵乘法
        try:
            encoder_seq = self.encoder(x, speaker_embedding)  # 编码输入文本
            encoder_seq_proj = self.encoder_proj(encoder_seq)  # 投影编码器输出
        except Exception as e:
            print(f"编码器处理时发生错误: {e}")
            # 创建一个合理的默认值
            encoder_seq = torch.zeros(batch_size, x.size(1), self.encoder_dims + self.speaker_embedding_size, device=device)
            encoder_seq_proj = torch.zeros(batch_size, x.size(1), self.decoder_dims, device=device)

        # 需要几个列表来存储输出
        mel_outputs, attn_scores, stop_outputs = [], [], []

        # 运行解码器循环
        for t in range(0, steps, self.r):
            try:
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame  # 使用前一个输出的最后一帧
                mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                            hidden_states, cell_states, context_vec, t, x)
                mel_outputs.append(mel_frames)  # 收集梅尔输出
                attn_scores.append(scores)  # 收集注意力分数
                stop_outputs.extend([stop_tokens] * self.r)  # 收集停止标记
                # 当批次中所有停止标记超过阈值且已过初始阶段时停止循环
                if (stop_tokens > 0.5).all() and t > 10: break
            except Exception as e:
                print(f"解码器循环中发生错误 (t={t}): {e}")
                # 创建一些合理的默认值
                mel_frames = torch.zeros(batch_size, self.n_mels, self.r, device=device)
                scores = torch.zeros(batch_size, 1, x.size(1), device=device)
                
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                # 如果连续多次出错，就退出循环
                if len(mel_outputs) >= 10 and t > 50:
                    print("连续多次错误，提前退出生成")
                    break

        # 将梅尔输出连接成序列
        try:
            mel_outputs = torch.cat(mel_outputs, dim=2)
        except Exception as e:
            print(f"连接梅尔输出时发生错误: {e}")
            # 创建一个默认的梅尔输出
            seq_len = min(len(mel_outputs) * self.r, steps)
            mel_outputs = torch.zeros(batch_size, self.n_mels, seq_len, device=device)

        # 后处理生成线性频谱图
        try:
            postnet_out = self.postnet(mel_outputs)  # 通过CBHG后处理网络
            linear = self.post_proj(postnet_out)  # 最终投影
            linear = linear.transpose(1, 2)  # 转置为(batch, time, dim)格式
        except Exception as e:
            print(f"后处理网络中发生错误: {e}")
            linear = torch.zeros(batch_size, mel_outputs.size(2), self.n_mels, device=device)

        # 方便可视化
        try:
            attn_scores = torch.cat(attn_scores, 1)  # 连接所有注意力分数
            stop_outputs = torch.cat(stop_outputs, 1)  # 连接所有停止标记
        except Exception as e:
            print(f"连接注意力分数或停止标记时发生错误: {e}")
            attn_scores = torch.zeros(batch_size, mel_outputs.size(2) // self.r, x.size(1), device=device)

        self.train()  # 恢复训练模式

        return mel_outputs, linear, attn_scores

    def init_model(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)  # 使用xavier均匀初始化多维参数

    def get_step(self):
        """获取当前训练步骤"""
        return self.step.data.item()

    def reset_step(self):
        """重置训练步骤"""
        # 对参数或缓冲区的赋值被重载，更新内部字典条目
        self.step = self.step.data.new_tensor(1)

    def log(self, path, msg):
        """记录消息到文件"""
        with open(path, "a") as f:
            print(msg, file=f)

    def load(self, path, optimizer=None):
        """加载模型(和优化器)状态"""
        try:
            # 检查模型是否有参数
            has_params = False
            for param in self.parameters():
                has_params = True
                device = param.device
                break
                
            if not has_params:
                print("警告：模型没有参数，无法加载权重")
                return False
                
            # 使用模型参数的设备作为加载位置
            checkpoint = torch.load(str(path), map_location=device)  # 加载检查点
            
            # 检查是否包含模型状态
            if "model_state" not in checkpoint:
                print(f"警告：检查点不包含model_state键，尝试直接加载")
                self.load_state_dict(checkpoint)
            else:
                self.load_state_dict(checkpoint["model_state"])  # 加载模型状态

            # 检查加载后是否有参数
            if sum(p.numel() for p in self.parameters()) == 0:
                print("警告：加载权重后模型仍没有参数")
                return False

            if "optimizer_state" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state"])  # 加载优化器状态
                
            print(f"模型已成功加载，参数数量: {sum(p.numel() for p in self.parameters()):,}")
            return True
        except Exception as e:
            print(f"加载模型时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save(self, path, optimizer=None):
        """保存模型(和优化器)状态"""
        try:
            # 检查模型是否有参数
            param_count = sum(p.numel() for p in self.parameters())
            if param_count == 0:
                print("警告：模型没有参数，无法保存权重")
                return False
                
            print(f"保存模型，参数数量: {param_count:,}")
            
            if optimizer is not None:
                torch.save({
                    "model_state": self.state_dict(),  # 保存模型状态
                    "optimizer_state": optimizer.state_dict(),  # 保存优化器状态
                    "step": self.step,  # 保存当前步数
                }, str(path))
            else:
                torch.save({
                    "model_state": self.state_dict(),  # 仅保存模型状态
                    "step": self.step,  # 保存当前步数
                }, str(path))
            
            print(f"模型已保存到: {path}")
            return True
        except Exception as e:
            print(f"保存模型时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def num_params(self, print_out=True):
        """计算和打印模型参数数量"""
        parameters = filter(lambda p: p.requires_grad, self.parameters())  # 只计算需要梯度的参数
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000  # 百万计数
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)  # 打印可训练参数数量
        return parameters
