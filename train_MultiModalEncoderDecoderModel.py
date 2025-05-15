# train_MultiModalEncoderDecoderModel.py

import torch
from data_loader_decoder import CustomDataset_decoder, collate_fn
from torchvision import transforms
import glob
import os
from collections import OrderedDict, defaultdict
import math
import random
from tqdm import tqdm  # 引入 tqdm 库
import time  # 引入 time 模块
import argparse  # 引入 argparse 模块
import sys
import numpy as np
sys.path.append('./imagebind_beam')
from imagebind.models.imagebind_model import ModalityType, ImageBindModel
from model import MultiModalEncoderDecoderModel
from decoder import MMWaveDecoder
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

# 设置seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # 如果使用的是 cudnn，以下设置可以进一步提高可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 选择一个固定的随机种子


# 定义自定义的 NMSE 损失函数
class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, output, target):
        mse = self.mse(output, target)
        var = torch.var(target, unbiased=False)
        # 防止除以零的情况
        if var.item() == 0:
            return torch.tensor(float('inf')).to(output.device)
        return mse / var

#---------新增topkloss---------
import torch.nn.functional as F

class TopkLoss(nn.Module):
    def __init__(self, k=1, reduction='mean'):
        super().__init__()
        self.k = k
        self.reduction = reduction

    def forward(self, output, target):
        """
        Args:
            output : [B, T, C] 模型输出的logits（未归一化）
            target : [B, T, C] one-hot编码 或 [B, T] 类别索引
        """
        # 转换target为类别索引
        if target.dim() == 3:
            target = torch.argmax(target, dim=-1)  # [B, T]
        
        B, T, C = output.shape
        output_flat = output.view(B*T, C)  # [B*T, C]
        target_flat = target.contiguous().view(-1)  # [B*T]
        
        # 计算Top-k正确性
        _, topk_indices = torch.topk(output_flat, self.k, dim=1)  # [B*T, k]
        correct = topk_indices.eq(target_flat.unsqueeze(1)).any(dim=1)  # [B*T]
        
        # 计算损失（仅惩罚Top-k错误的样本）
        loss = F.cross_entropy(output_flat, target_flat, reduction='none')  # [B*T]
        masked_loss = loss * ~correct  # 错误样本保留损失
        
        if self.reduction == 'mean':
            return masked_loss.mean()
        elif self.reduction == 'sum':
            return masked_loss.sum()
        return masked_loss

#------------------------------

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train MultiModalEncoderDecoderModel with MSE or NMSE loss.')
    parser.add_argument('--loss', type=str, choices=['MSE', 'NMSE','TOPK'], default='MSE',
                        help='选择损失函数类型：MSE、NMSE 或 TOPK。默认是 MSE。')
    parser.add_argument('--epochs', type=int, default=50, help='训练的总轮数。默认是50。')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小。默认是16。')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率。默认是1e-4。')
    parser.add_argument('--patience', type=int, default=10, help='早停步数。')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型检查点保存目录。默认是./checkpoints。')
    parser.add_argument('--pretrained_encoder_path', type=str, required=True,
                        help='预训练编码器权重的路径。')
    parser.add_argument('--dataset_start_idx', type=int, default=1, help='数据集起始索引。默认是1。')
    parser.add_argument('--dataset_end_idx', type=int, default=9, help='数据集结束索引。默认是9。')
    return parser.parse_args()

def split_dataset_per_scenario_decoder(dataset, test_size=0.1, val_size=0.1, min_samples=10, random_state=42):
    """
    按场景内部划分训练集、验证集和测试集。
    要求每个场景的样本数量大于等于 min_samples，按照 80%:10%:10% 划分。
    验证集和测试集样本数向下取整，训练集使用剩余所有样本。

    Args:
        dataset (CustomDataset_decoder): 自定义数据集。
        test_size (float): 测试集比例（相对于每个场景的样本数）。
        val_size (float): 验证集比例（相对于每个场景的样本数）。
        min_samples (int): 每个场景最小样本数。
        random_state (int): 随机种子。

    Returns:
        train_indices (list): 训练集滑动窗口样本索引列表。
        val_indices (list): 验证集滑动窗口样本索引列表。
        test_indices (list): 测试集滑动窗口样本索引列表。
    """
    # 按场景分组滑动窗口样本索引
    scenario_to_window_indices = defaultdict(list)
    for idx, scenario in enumerate(dataset.scenario_list):
        scenario_to_window_indices[scenario].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    # 遍历每个场景，进行划分
    for scenario, window_indices in scenario_to_window_indices.items():
        n_samples = len(window_indices)

        if n_samples < min_samples:
            # 如果场景内样本数不足 min_samples，全部分配到训练集
            train_indices.extend(window_indices)
            print(f"Scenario '{scenario}' has {n_samples} samples; all assigned to train set.")
            continue

        # 计算验证集和测试集的样本数
        val_count = math.floor(n_samples * val_size)
        test_count = math.floor(n_samples * test_size)
        train_count = n_samples - val_count - test_count  # 剩余样本分配到训练集

        # 打乱样本顺序
        shuffled = window_indices.copy()
        random_seed = random_state
        random.Random(random_seed).shuffle(shuffled)

        # 分配样本
        test_scenario_indices = shuffled[:test_count]
        val_scenario_indices = shuffled[test_count:test_count + val_count]
        train_scenario_indices = shuffled[test_count + val_count:]

        # 分别添加到各自的列表中
        train_indices.extend(train_scenario_indices)
        val_indices.extend(val_scenario_indices)
        test_indices.extend(test_scenario_indices)

        print(f"Scenario '{scenario}': {len(train_scenario_indices)} train, {len(val_scenario_indices)} val, {len(test_scenario_indices)} test.")

    return train_indices, val_indices, test_indices

def main():
    # 解析命令行参数
    args = parse_args()

    # 1. 数据加载和划分部分

    # 定义转换，与预训练时保持一致
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 与预训练时一致
                             std=[0.229, 0.224, 0.225]),
    ])

    # 定义数据集路径
    dataset_path = [f'/data2/wzj/Datasets/DeepSense/scenario{i}/' for i in range(args.dataset_start_idx, args.dataset_end_idx)]  # scenario1 ~ scenario8
    # dataset_path = [f'/new_disk/yy/DeepSensePre/Data_raw/scenario{i}/' for i in range(2, 3)]  # scenario1 ~ scenario1
    data_csv_paths = []
    for path in dataset_path:
        data_csv_paths.extend(glob.glob(os.path.join(path, '*.csv')))

    print(f"Found {len(data_csv_paths)} CSV files for training.")

    # 初始化 CustomDataset_decoder
    modal = 'mmwave_gps'  # 选择模态
    input_length = 8
    output_length = 3
    dataset = CustomDataset_decoder(
        data_csv_paths, 
        transform=default_transform, 
        modal=modal, 
        input_length=input_length, 
        output_length=output_length
    )

    # # 按场景内部划分训练集、验证集和测试集
    # train_indices, val_indices, test_indices = split_dataset_per_scenario_decoder(
    #     dataset, 
    #     test_size=0.1, 
    #     val_size=0.1, 
    #     min_samples=10, 
    #     random_state=42
    # )
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"Total Training samples: {len(train_dataset)}")
    print(f"Total Validation samples: {len(val_dataset)}")
    print(f"Total Testing samples: {len(test_dataset)}")

    # # 创建 Subset
    # train_dataset = Subset(dataset, train_indices)
    # val_dataset = Subset(dataset, val_indices)
    # test_dataset = Subset(dataset, test_indices)

    # 创建 DataLoader
    batch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 打乱训练集
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 2. 模型加载部分

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化编码器
    encoder = ImageBindModel(
        video_frames=input_length,  # 与数据加载时的 input_length 一致
        # 其他参数应与预训练时保持一致
    ).to(device)

    # 加载预训练的编码器权重
    model_save_path = args.pretrained_encoder_path  # 使用命令行参数提供的路径
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Pretrained encoder weights not found at {model_save_path}")

    state_dict = torch.load(model_save_path, map_location=device)
    print(f"Loaded state_dict from {model_save_path}")

    # 处理多GPU保存的模型（如果适用）
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 加载权重到编码器
    encoder.load_state_dict(new_state_dict)
    print("Encoder weights loaded successfully.")

    # 设置编码器为冻结（如果不需要微调）
    for param in encoder.parameters():
        param.requires_grad = False

    # 初始化解码器
    # decoder = MMWaveDecoder(
    #     embed_dim=768,        # 根据 decoder 的定义
    #     nhead=8,
    #     num_layers=6,
    #     dim_feedforward=2048,
    #     dropout=0.1,
    #     output_dim=64,        # mmWave 数据的维度，根据实际情况调整
    #     max_seq_length=output_length
    # ).to(device)
    decoder = MMWaveDecoder(
        embed_dim=768,        # 根据 decoder 的定义
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        dropout=0.1,
        output_dim=64,        # mmWave 数据的维度，根据实际情况调整
        max_seq_length=output_length
    ).to(device)

    # 初始化集成模型，提供各模态的特征维度
    vision_dim = 768  # 假设 VISION 特征维度为 768
    gps_dim = 768     # 假设 GPS 特征维度为 768
    mmwave_dim = 768  # 确保 mmwave_dim 设置为 768

    # 初始化集成模型
    model = MultiModalEncoderDecoderModel(
        encoder=encoder, 
        decoder=decoder, 
        vision_dim=vision_dim, 
        gps_dim=gps_dim, 
        mmwave_dim=mmwave_dim,
        input_length=input_length,
        output_length=output_length
    ).to(device)

    # 3. 定义损失函数和优化器

    # 根据选择的损失函数设置 criterion
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
        print("Using MSELoss as the loss function.")
    elif args.loss == 'NMSE':
        criterion = NMSELoss()
        print("Using NMSELoss as the loss function.")
    #----------添加topkloss----------
    elif args.loss == 'topkloss':
        criterion = TopkLoss(k=3, reduction='mean')
        print("Using TopkLoss as the loss function.")
    #--------------------------------
    else:
        raise ValueError(f"Unsupported loss type: {args.loss}")

    # 优化器仅优化解码器的参数
    optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)

    # 4. 定义学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True
    )

    # 5. 定义训练和验证函数

    def train_one_epoch(model, data_loader, criterion, optimizer, device):
        model.train()
        epoch_loss = 0.0
        # 使用 tqdm 进度条包装 data_loader
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Training', leave=False)):
            # 准备输入
            inputs = {}
            if modal == 'gps':
                inputs[ModalityType.VISION] = batch['video'].to(device)
                inputs[ModalityType.GPS] = batch['gps'].to(device)
            elif modal == 'mmwave':
                inputs[ModalityType.VISION] = batch['video'].to(device)
                inputs[ModalityType.MMWAVE] = batch['mmwave'].to(device)
            elif modal == 'mmwave_gps':
                inputs[ModalityType.VISION] = batch['video'].to(device)
                inputs[ModalityType.GPS] = batch['gps'].to(device)
                inputs[ModalityType.MMWAVE] = batch['mmwave'].to(device)
            else:
                raise ValueError(f"Unsupported modal: {modal}")

            target_mmwave = batch['target_mmwave'].to(device)  # [B, output_length, D]

            # 准备目标序列的输入（shifted right by one for teacher forcing）
            # 在这里，我们使用全零作为目标序列的起始输入
            tgt_seq = torch.zeros(target_mmwave.size(0), output_length, dtype=torch.long, device=device)  # [B, output_length]

            # 前向传播
            optimizer.zero_grad()
            output = model(inputs, tgt_seq)  # [B, output_length, D]

            # 计算损失
            loss = criterion(output, target_mmwave)
            epoch_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(data_loader)
        return avg_loss

    def evaluate(model, data_loader, criterion, device):
        model.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='Evaluating', leave=False)):
                # 准备输入
                inputs = {}
                if modal == 'gps':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.GPS] = batch['gps'].to(device)
                elif modal == 'mmwave':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.MMWAVE] = batch['mmwave'].to(device)
                elif modal == 'mmwave_gps':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.GPS] = batch['gps'].to(device)
                    inputs[ModalityType.MMWAVE] = batch['mmwave'].to(device)
                else:
                    raise ValueError(f"Unsupported modal: {modal}")

                target_mmwave = batch['target_mmwave'].to(device)  # [B, output_length, D]

                # 准备目标序列的输入
                tgt_seq = torch.zeros(target_mmwave.size(0), output_length, dtype=torch.long, device=device)  # [B, output_length]

                # 前向传播
                output = model(inputs, tgt_seq)  # [B, output_length, D]

                # 计算损失
                loss = criterion(output, target_mmwave)
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        return avg_loss

    # 6. 训练循环

    def format_time(seconds):
        mins, sec = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f"{int(hrs)}h {int(mins)}m {int(sec)}s"

    num_epochs = args.epochs
    best_val_loss = float('inf')

    # 确保保存模型的目录存在
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 记录训练开始时间
    training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # 计算剩余时间
        elapsed_time = epoch_end_time - training_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        remaining_time = avg_epoch_time * remaining_epochs

        # 转换为更易读的格式
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch Duration: {format_time(epoch_duration)}, Estimated Remaining Time: {format_time(remaining_time)}")

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.checkpoint_dir, 'multimodal_encoder_decoder_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch+1} to {best_model_path}")
            early_stop_counter = 0  # 重置计数器
        else:
            early_stop_counter += 1  # 增加计数器

        # 如果验证损失连续多个 epoch 没有改善，则停止训练
        if early_stop_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break  # 提前停止训练

        # 每隔若干个 epoch 保存模型
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'multimodal_encoder_decoder_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model at epoch {epoch+1} to {checkpoint_path}")

    # 7. 测试评估

    # 加载最佳模型
    best_model_path = os.path.join(args.checkpoint_dir, 'multimodal_encoder_decoder_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        print("Loaded best model for testing.")
    else:
        print(f"Best model not found at {best_model_path}. Skipping test evaluation.")

    # 定义测试评估函数（可以与验证相同）
    def test_evaluate(model, data_loader, criterion, device):
        model.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='Testing', leave=False)):
                # 准备输入
                inputs = {}
                if modal == 'gps':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.GPS] = batch['gps'].to(device)
                elif modal == 'mmwave':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.MMWAVE] = batch['mmwave'].to(device)
                elif modal == 'mmwave_gps':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.GPS] = batch['gps'].to(device)
                    inputs[ModalityType.MMWAVE] = batch['mmwave'].to(device)
                else:
                    raise ValueError(f"Unsupported modal: {modal}")

                target_mmwave = batch['target_mmwave'].to(device)  # [B, output_length, D]

                # 准备目标序列的输入
                tgt_seq = torch.zeros(target_mmwave.size(0), output_length, dtype=torch.long, device=device)  # [B, output_length]

                # 前向传播
                output = model(inputs, tgt_seq)  # [B, output_length, D]

                # 计算损失
                loss = criterion(output, target_mmwave)
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        return avg_loss

    test_loss = test_evaluate(model, test_loader, criterion, device)
    print(f"Test Loss ({args.loss}): {test_loss:.4f}")

if __name__ == '__main__':
    main()
