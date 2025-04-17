# ResNet语音编码器

本项目实现了基于ResNet架构的语音编码器，用于替代原始语音克隆系统中的LSTM编码器。ResNet编码器具有更强的特征提取能力，可以更好地捕捉语音中的声纹特征。

## 主要特点

- 使用ResNet50架构，适合语音特征提取
- 包含嵌入空间调整层，使输出更接近原始LSTM编码器
- 保持与原始SpeakerEncoder类相同的接口
- 支持与现有合成器的兼容性

## 文件结构

- `encoder_adapter.py`: ResNet编码器的核心实现
- `encoder_train_resnet.py`: ResNet编码器的训练脚本
- `integrate_resnet.py`: 将ResNet编码器集成到语音克隆系统的脚本
- `saved_models/`: 保存训练好的模型权重

## 使用方法

### 1. 训练ResNet编码器

```bash
python encoder_train_resnet.py resnet_training ./dataset/SV2TTS/encoder -m saved_models
```

### 2. 训练适配层（推荐）

为了确保ResNet编码器与现有合成器的兼容性，建议训练一个适配层：

```bash
python train_resnet_adapter.py adapter_training ./dataset/SV2TTS/encoder \
    --lstm_weights_path saved_models/default/encoder.pt \
    --resnet_weights_path ResNet/saved_models/resnet_encoder.pt \
    -b 64 -lr 1e-4 -s 10000 --save_every 1000 --eval_every 100
```

### 3. 使用ResNet编码器进行语音合成

```bash
python integrate_resnet.py --weights_path ResNet/saved_models/resnet_adapter.pt
```

## 技术细节

### ResNet编码器架构

ResNet编码器使用ResNet50架构，包括：

1. 初始卷积层：7x7卷积，步长为2，64通道
2. 四个残差块组，通道数分别为256、512、1024和2048
3. 全局平均池化
4. 全连接层，输出256维嵌入向量
5. 嵌入空间调整层，使输出更接近LSTM编码器

### 嵌入空间调整

为了确保ResNet编码器与现有合成器的兼容性，我们添加了一个嵌入空间调整层：

```python
self.embedding_adapter = nn.Sequential(
    nn.Linear(model_embedding_size, model_embedding_size),
    nn.Tanh(),
    nn.Linear(model_embedding_size, model_embedding_size)
)
```

这个层通过训练，将ResNet的嵌入向量映射到更接近LSTM编码器的表示空间。

### 适配层训练

适配层训练使用两种损失函数：

1. MSE损失：确保嵌入向量的绝对位置接近
2. 余弦相似度损失：确保嵌入向量的方向接近

训练过程中，我们冻结ResNet编码器的参数，只训练适配层。

## 与原始LSTM编码器的区别

1. **架构差异**：
   - LSTM编码器：使用LSTM处理时序特征
   - ResNet编码器：使用卷积和残差连接处理空间特征

2. **特征提取能力**：
   - LSTM编码器：擅长捕捉时序依赖关系
   - ResNet编码器：擅长捕捉局部特征和层次结构

3. **训练稳定性**：
   - LSTM编码器：对学习率敏感，需要仔细调整
   - ResNet编码器：训练更稳定，可以使用更大的学习率

4. **计算效率**：
   - LSTM编码器：计算复杂度与序列长度成正比
   - ResNet编码器：计算复杂度与输入大小成正比，通常更高效

## 注意事项

1. 首次使用时，建议先训练适配层，确保与现有合成器的兼容性
2. 如果适配层效果不理想，可以考虑重新训练合成器
3. 适配层训练需要原始LSTM编码器的权重文件

## 未来改进

1. 探索更复杂的适配层架构
2. 尝试端到端训练ResNet编码器和合成器
3. 研究其他网络架构，如Transformer等 