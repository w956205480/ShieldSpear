from utils.argutils import print_args
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from ResNet.encoder_adapter import ResNetSpeakerEncoder
from encoder.params_model import *
from encoder.visualizations import Visualizations
from utils.profiler import Profiler
from pathlib import Path
import argparse
import torch
import os

# 为Visualizations类添加update_val方法
if not hasattr(Visualizations, 'update_val'):
    def update_val(self, val_loss, val_eer, step):
        """更新验证集上的指标"""
        print(f"\nValidation - Step {step}: Loss: {val_loss:.4f}, EER: {val_eer:.4f}")
        
        if self.disabled:
            return
        
        if hasattr(self, 'vis'):
            try:
                # 创建验证损失图
                val_loss_win = getattr(self, 'val_loss_win', None)
                self.val_loss_win = self.vis.line(
                    [val_loss],
                    [step],
                    win=val_loss_win,
                    update="append" if val_loss_win else None,
                    opts=dict(
                        legend=["Validation loss"],
                        xlabel="Step",
                        ylabel="Loss",
                        title="Validation Loss",
                    )
                )
                
                # 创建验证EER图
                val_eer_win = getattr(self, 'val_eer_win', None)
                self.val_eer_win = self.vis.line(
                    [val_eer],
                    [step],
                    win=val_eer_win,
                    update="append" if val_eer_win else None,
                    opts=dict(
                        legend=["Validation EER"],
                        xlabel="Step",
                        ylabel="EER",
                        title="Validation Equal error rate"
                    )
                )
            except Exception as e:
                print(f"可视化更新失败: {e}")
    
    # 添加方法到类
    Visualizations.update_val = update_val
    print("已添加update_val方法到Visualizations类")

#python encoder_train_resnet.py resnet_training_run ./dataset/SV2TTS/encoder -m saved_models -s 1000 -u 100 -v 10 --no_visdom
def sync(device: torch.device):
    """同步设备，确保正确计时（CUDA操作是异步的）"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_resnet_encoder(run_id: str, clean_data_root: Path, test_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
                      backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
                      no_visdom: bool, max_steps: int = None):
    """
    训练ResNet说话者编码器
    
    参数:
        run_id: 训练运行的唯一标识符
        clean_data_root: 预处理后的训练数据根目录
        test_data_root: 预处理后的测试数据根目录
        models_dir: 模型保存目录
        umap_every: 每多少步更新一次UMAP投影
        save_every: 每多少步保存一次模型
        backup_every: 每多少步备份一次模型
        vis_every: 每多少步更新一次可视化
        force_restart: 是否强制重新开始训练
        visdom_server: Visdom服务器地址
        no_visdom: 是否禁用Visdom
        max_steps: 最大训练步数，默认为None（一直训练）
    """
    # 创建训练数据集和数据加载器
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=4,
    )

    # 创建测试数据集和数据加载器
    test_dataset = SpeakerVerificationDataset(test_data_root)
    test_loader = SpeakerVerificationDataLoader(
        test_dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=2,
    )
    print(f"训练数据集: {len(dataset.speakers)}位说话人")
    print(f"测试数据集: {len(test_dataset.speakers)}位说话人")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    # 创建ResNet编码器模型和优化器
    model = ResNetSpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1

    # 配置模型文件路径
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "resnet_encoder.pt"

    # 加载现有模型（如果存在）
    if not force_restart:
        if state_fpath.exists():
            print("找到现有模型 \"%s\"，加载并继续训练。" % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            
            # 兼容性处理：加载旧版模型权重
            try:
                model.load_state_dict(checkpoint["model_state"])
            except RuntimeError as e:
                print(f"警告：加载模型权重时出现问题: {e}")
                print("正在尝试部分加载模型权重...")
                
                # 获取当前模型的状态字典
                model_state = model.state_dict()
                
                # 加载存在的键
                for key in checkpoint["model_state"]:
                    if key in model_state:
                        model_state[key] = checkpoint["model_state"][key]
                        print(f"成功加载参数: {key}")
                
                # 手动加载状态字典，忽略缺失的键
                model.load_state_dict(model_state, strict=False)
                print("部分加载完成，缺失的参数将使用初始化值。")
            
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("没有找到模型 \"%s\"，从头开始训练。" % run_id)
    else:
        print("强制从头开始训练。")
    model.train()

    # 初始化可视化环境
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name, "Model": "ResNet"})

    # 训练循环
    profiler = Profiler(summarize_every=10, disabled=False)
    
    # 创建保存目录的ResNet/saved_models子目录
    resnet_models_dir = Path("ResNet/saved_models")
    resnet_models_dir.mkdir(exist_ok=True, parents=True)
    resnet_state_fpath = resnet_models_dir / "resnet_encoder.pt"

    # 存储最佳验证损失和最佳验证EER
    best_val_loss = float('inf')
    best_val_eer = float('inf')
    
    for step, speaker_batch in enumerate(loader, init_step):
        profiler.tick("等待批次数据（线程阻塞）")
        
        # 检查是否达到最大步数
        if max_steps is not None and step > max_steps:
            print(f"已达到最大训练步数 {max_steps}，停止训练")
            break

        # 前向传播
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("数据转移到 %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("前向传播")
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("损失计算")

        # 反向传播
        model.zero_grad()
        loss.backward()
        profiler.tick("反向传播")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("参数更新")

        # 更新可视化
        vis.update(loss.item(), eer, step)

        # 绘制投影并保存
        if umap_every != 0 and step % umap_every == 0:
            print("绘制并保存投影（步骤 %d）" % step)
            projection_fpath = model_dir / f"umap_{step:06d}.png"
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
            vis.save()

        # 在验证集上评估，并保存模型
        if save_every != 0 and step % save_every == 0:
            # 在验证集上评估
            model.eval()
            val_losses = []
            val_eers = []
            
            print(f"在验证集上评估（步骤 {step}）...")
            with torch.no_grad():
                for val_batch_idx, val_speaker_batch in enumerate(test_loader):
                    if val_batch_idx >= 10:  # 限制验证批次数量以节省时间
                        break
                    val_inputs = torch.from_numpy(val_speaker_batch.data).to(device)
                    val_embeds = model(val_inputs)
                    val_embeds_loss = val_embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
                    val_loss, val_eer = model.loss(val_embeds_loss)
                    val_losses.append(val_loss.item())
                    val_eers.append(val_eer)
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_eer = sum(val_eers) / len(val_eers)
            print(f"验证集结果 - 损失: {avg_val_loss:.4f}, EER: {avg_val_eer:.4f}")
            
            # 记录验证指标
            vis.update_val(avg_val_loss, avg_val_eer, step)
            
            # 切回训练模式
            model.train()
            
            # 保存当前模型
            print("保存模型（步骤 %d）" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_eer": avg_val_eer,
            }, state_fpath)
            
            # 同时保存到ResNet/saved_models目录
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_eer": avg_val_eer,
            }, resnet_state_fpath)
            
            # 如果验证损失更好，保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_eer = avg_val_eer
                best_fpath = model_dir / "resnet_encoder_best.pt"
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_eer": avg_val_eer,
                }, best_fpath)
                print(f"保存最佳模型 - 验证损失: {avg_val_loss:.4f}, EER: {avg_val_eer:.4f}")

        # 创建备份
        if backup_every != 0 and step % backup_every == 0:
            print("创建备份（步骤 %d）" % step)
            backup_fpath = model_dir / f"resnet_encoder_{step:06d}.bak"
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)

        profiler.tick("附加操作（可视化、保存）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="训练ResNet说话者编码器。需要先运行encoder_preprocess.py。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("run_id", type=str, help= \
        "模型名称。默认情况下，训练输出将保存到saved_models/<run_id>/。如果之前已保存相同run_id的模型状态，"
        "训练将从那里继续。使用-f可以覆盖已保存的状态并从头开始。")
    parser.add_argument("clean_data_root", type=Path, help= \
        "encoder_preprocess.py的训练数据输出目录路径。应为dataset/SV2TTS/encoder/。")
    parser.add_argument("--test_data_root", type=Path, default=None, help= \
        "encoder_preprocess.py的测试数据输出目录路径。默认为dataset/SV2TTS_test/encode/。")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", help=\
        "包含所有模型的根目录路径。将在此根目录下创建<run_name>目录。"
        "它将包含保存的模型权重，以及这些权重的备份和训练期间生成的图表。")
    parser.add_argument("-v", "--vis_every", type=int, default=10, help= \
        "更新损失和图表之间的步数。")
    parser.add_argument("-u", "--umap_every", type=int, default=100, help= \
        "更新umap投影之间的步数。设置为0表示永不更新投影。")
    parser.add_argument("-s", "--save_every", type=int, default=500, help= \
        "在磁盘上更新模型之间的步数。设置为0表示永不保存模型。")
    parser.add_argument("-b", "--backup_every", type=int, default=7500, help= \
        "模型备份之间的步数。设置为0表示永不备份模型。")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "不加载任何已保存的模型，从头开始训练。")
    parser.add_argument("--visdom_server", type=str, default="http://localhost")
    parser.add_argument("--no_visdom", action="store_true", help= \
        "禁用visdom可视化。")
    parser.add_argument("--max_steps", type=int, default=None, help= \
        "最大训练步数，默认为None（一直训练直到手动停止）。")
    args = parser.parse_args()

    # 设置默认的测试数据路径（如果未指定）
    if args.test_data_root is None:
        args.test_data_root = Path("dataset/SV2TTS_test/encode")
    
    # 检查路径是否存在
    if not args.clean_data_root.exists():
        raise FileNotFoundError(f"训练数据目录不存在: {args.clean_data_root}")
    if not args.test_data_root.exists():
        raise FileNotFoundError(f"测试数据目录不存在: {args.test_data_root}")

    # 运行训练
    print_args(args, parser)
    train_resnet_encoder(**vars(args)) 