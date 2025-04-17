import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# 导入所需的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoder.params_model import *
from encoder.params_data import *

class ResidualBlock(nn.Module):
    """
    ResNet残差块
    使用瓶颈结构（bottleneck）设计，包含1x1、3x3、1x1卷积
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(ResidualBlock, self).__init__()
        self.expansion = expansion
        mid_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)
        
        # 如果输入输出通道数不同，需要1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * expansion)
            )
        
        # 添加SE注意力模块
        self.se = SELayer(out_channels * expansion)
        
        # 添加Dropout
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # 应用SE注意力
        out = self.se(out)
        
        # 添加残差连接
        out += self.shortcut(residual)
        out = F.relu(out)
        
        # 应用Dropout
        out = self.dropout(out)
        
        return out

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation注意力模块
    通过全局平均池化和全连接层学习通道注意力
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNetSpeakerEncoder(nn.Module):
    """
    ResNet说话者编码器，替代原有的LSTM编码器
    该类保持与原SpeakerEncoder类相同的接口，但使用ResNet模型进行特征提取
    """
    
    def __init__(self, device=None, loss_device=None):
        """
        初始化ResNet说话者编码器
        
        参数:
            device: 模型计算设备，默认为GPU（如果可用）
            loss_device: 损失计算设备，默认为CPU
        """
        super().__init__()
        
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        if loss_device is None:
            loss_device = torch.device("cpu")
        self.loss_device = loss_device
        
        # 创建ResNet模型
        self.create_resnet_model()
        
        # 相似度计算参数（与原SpeakerEncoder一致）
        self.similarity_weight = nn.Parameter(torch.tensor([10.], device=device))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.], device=device))
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
        # 新增：嵌入空间调整层
        # 这个层将ResNet的嵌入向量映射到更接近LSTM的嵌入空间
        self.embedding_adapter = nn.Sequential(
            nn.Linear(model_embedding_size, model_embedding_size),
            nn.Tanh(),
            nn.Linear(model_embedding_size, model_embedding_size)
        ).to(device)
        
        # 将模型移动到指定设备
        self.to(device)
    
    def create_resnet_model(self):
        """
        创建完整的ResNet模型
        使用ResNet50架构，适合语音特征提取
        """
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ).to(self.device)
        
        # ResNet块 - 使用ResNet50架构
        self.layer1 = self._make_layer(64, 64, 3, stride=1)  # 256通道
        self.layer2 = self._make_layer(256, 128, 4, stride=2)  # 512通道
        self.layer3 = self._make_layer(512, 256, 6, stride=2)  # 1024通道
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)  # 2048通道
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        
        # 全连接层，输出嵌入向量
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, model_embedding_size)
        ).to(self.device)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        创建包含多个残差块的层
        """
        layers = []
        # 第一个块可能需要调整步长
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # 其余块步长为1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels * 4, out_channels, 1))
        return nn.Sequential(*layers).to(self.device)
    
    def forward(self, utterances, hidden_init=None):
        """
        计算一批语音梅尔频谱图的嵌入向量
        
        参数:
            utterances: 形状为(batch_size, n_frames, n_channels)的梅尔频谱图批次
            hidden_init: 为了与原接口兼容，这里不使用
        
        返回:
            形状为(batch_size, embedding_size)的嵌入向量
        """
        # 添加通道维度，转为(batch_size, 1, n_frames, n_channels)
        x = utterances.unsqueeze(1)
        
        # 通过ResNet层提取特征
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 通过全连接层生成嵌入向量
        embeds_raw = self.fc(x)
        
        # L2归一化
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)
        
        # 新增：通过嵌入空间调整层 - 使ResNet输出更接近LSTM编码器的表示空间
        adapted_embeds = self.embedding_adapter(embeds)
        
        # 再次进行L2归一化，确保输出向量范数为1
        adapted_embeds = adapted_embeds / (torch.norm(adapted_embeds, dim=1, keepdim=True) + 1e-5)
        
        return adapted_embeds
    
    def similarity_matrix(self, embeds):
        """
        计算相似度矩阵，与原SpeakerEncoder类保持一致的实现
        
        参数:
            embeds: 形状为(speakers_per_batch, utterances_per_speaker, embedding_size)的嵌入向量
        
        返回:
            相似度矩阵
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # 确保embeds在正确的设备上
        embeds = embeds.to(self.device)
        
        # 包含质心（每个说话人一个）
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)
        
        # 排他质心（每个话语一个）
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)
        
        # 相似矩阵计算 - 确保在正确的设备上
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                               speakers_per_batch).to(self.device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int_)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        # 直接使用self.similarity_weight和self.similarity_bias
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        
        # 最后将结果移动到loss_device以计算损失
        sim_matrix = sim_matrix.to(self.loss_device)
        return sim_matrix
    
    def loss(self, embeds):
        """
        计算损失和EER，与原SpeakerEncoder类保持一致的实现
        
        参数:
            embeds: 形状为(speakers_per_batch, utterances_per_speaker, embedding_size)的嵌入向量
        
        返回:
            损失值和EER
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # 确保embeds在正确的设备上
        embeds = embeds.to(self.device)
        
        # 计算损失
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                       speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        # 计算EER（不参与反向传播）
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int_)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()
            
            # ROC曲线计算
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer
    
    def do_gradient_ops(self):
        """
        执行梯度操作，与原SpeakerEncoder类一致
        """
        # 使用参数字典遍历参数处理梯度，避免直接访问非叶张量的梯度
        for param in self.parameters():
            if param.grad is not None:
                if param is self.similarity_weight:
                    param.grad *= 0.01
                elif param is self.similarity_bias:
                    param.grad *= 0.01
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), 3, norm_type=2)

def load_resnet_encoder(weights_fpath, device=None):
    """
    加载ResNet编码器的预训练权重
    
    参数:
        weights_fpath: 预训练权重文件路径
        device: 计算设备
    
    返回:
        加载好权重的ResNet编码器
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建编码器
    encoder = ResNetSpeakerEncoder(device)
    
    # 加载权重
    checkpoint = torch.load(weights_fpath, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state'])
    else:
        encoder.load_state_dict(checkpoint)
    
    # 设置为评估模式
    encoder.eval()
    return encoder

def replace_encoder():
    """
    替换原有编码器为ResNet编码器，通过猴子补丁实现
    
    该函数将全局替换encoder.model.SpeakerEncoder类为ResNetSpeakerEncoder类
    """
    import encoder.model
    # 保存原始类以便日后可能的恢复
    encoder.model._OriginalSpeakerEncoder = encoder.model.SpeakerEncoder
    # 替换为ResNet编码器
    encoder.model.SpeakerEncoder = ResNetSpeakerEncoder
    print("已将原始SpeakerEncoder替换为ResNetSpeakerEncoder")

def setup_inference_encoder(weights_fpath=None):
    """
    设置用于推理的ResNet编码器
    
    参数:
        weights_fpath: 预训练权重文件路径，如果为None则使用默认路径
    
    返回:
        设置好的ResNet编码器
    """
    if weights_fpath is None:
        weights_fpath = Path("ResNet/saved_models/resnet_encoder.pt")
    else:
        weights_fpath = Path(weights_fpath)
    
    if not weights_fpath.exists():
        print(f"警告：找不到ResNet编码器权重文件 {weights_fpath}")
        print("使用随机初始化的ResNet编码器，性能可能不佳")
        return ResNetSpeakerEncoder()
    
    return load_resnet_encoder(weights_fpath)

def update_inference_module(weights_fpath=None):
    """
    更新推理模块，使用ResNet编码器替代原始编码器
    
    参数:
        weights_fpath: ResNet编码器权重文件路径，如果为None则使用默认路径
    """
    # 替换编码器类
    replace_encoder()
    
    # 设置推理编码器
    encoder = setup_inference_encoder(weights_fpath)
    
    # 更新推理模块
    import encoder.inference
    encoder.inference.encoder = encoder
    print("已更新推理模块，使用ResNet编码器") 