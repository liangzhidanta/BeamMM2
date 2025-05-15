# train_MMWaveEncoderDecoderModel.py

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
sys.path.append('./imagebind_beam')
from imagebind.models.imagebind_model import ModalityType, ImageBindModel
from model import RawMMWaveDecoderOnlyModel
from decoder import MMWaveDecoder
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

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

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train MMWaveEncoderDecoderModel with MSE or NMSE loss.')
    parser.add_argument('--loss', type=str, choices=['MSE', 'NMSE'], default='MSE',
                        help='选择损失函数类型：MSE 或 NMSE。默认是 MSE。')
    parser.add_argument('--epochs', type=int, default=50, help='训练的总轮数。默认是50。')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小。默认是16。')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率。默认是1e-4。')
    parser.add_argument('--patience', type=int, default=10, help='早停步数。')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型检查点保存目录。默认是./checkpoints。')
    parser.add_argument('--pretrained_encoder_path', type=str, required=True,
                        help='预训练编码器权重的路径。')
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
    dataset_path = [f'/new_disk/yy/DeepSensePre/Data_raw/scenario{i}/' for i in range(1, 9)]  # scenario1 ~ scenario9
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

    # 按场景内部划分训练集、验证集和测试集
    train_indices, val_indices, test_indices = split_dataset_per_scenario_decoder(
        dataset, 
        test_size=0.1, 
        val_size=0.1, 
        min_samples=10, 
        random_state=42
    )

    print(f"Total Training samples: {len(train_indices)}")
    print(f"Total Validation samples: {len(val_indices)}")
    print(f"Total Testing samples: {len(test_indices)}")

    # 创建 Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    raw_mmwave_dim = 64  # 原始mmWave数据维度

    # 初始化集成模型
    model = RawMMWaveDecoderOnlyModel(
        decoder=decoder,  
        raw_mmwave_dim=raw_mmwave_dim,
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
            raw_mmwave = batch['mmwave'].to(device)  # [B, input_length, D_raw]

            target_mmwave = batch['target_mmwave'].to(device)  # [B, output_length, D]

            # 准备目标序列的输入（shifted right by one for teacher forcing）
            # 在这里，我们使用全零作为目标序列的起始输入
            tgt_seq = torch.zeros(target_mmwave.size(0), output_length, dtype=torch.long, device=device)  # [B, output_length]

            # 前向传播
            optimizer.zero_grad()
            output = model(raw_mmwave, tgt_seq)  # [B, output_length, D]

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
                raw_mmwave = batch['mmwave'].to(device)  # [B, input_length, D_raw]

                target_mmwave = batch['target_mmwave'].to(device)  # [B, output_length, D]

                # 准备目标序列的输入
                tgt_seq = torch.zeros(target_mmwave.size(0), output_length, dtype=torch.long, device=device)  # [B, output_length]

                # 前向传播
                output = model(raw_mmwave, tgt_seq)  # [B, output_length, D]

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
            best_model_path = os.path.join(args.checkpoint_dir, 'rawmmwave_decoder_best.pth')
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
            checkpoint_path = os.path.join(args.checkpoint_dir, f'rawmmwave_decoder_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model at epoch {epoch+1} to {checkpoint_path}")

    # 7. 测试评估

    # 加载最佳模型
    best_model_path = os.path.join(args.checkpoint_dir, 'rawmmwave_decoder_best.pth')
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
                raw_mmwave = batch['mmwave'].to(device)  # [B, input_length, D_raw]

                target_mmwave = batch['target_mmwave'].to(device)  # [B, output_length, D]

                # 准备目标序列的输入
                tgt_seq = torch.zeros(target_mmwave.size(0), output_length, dtype=torch.long, device=device)  # [B, output_length]

                # 前向传播
                output = model(raw_mmwave, tgt_seq)  # [B, output_length, D]

                # 计算损失
                loss = criterion(output, target_mmwave)
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        return avg_loss

    test_loss = test_evaluate(model, test_loader, criterion, device)
    print(f"Test Loss ({args.loss}): {test_loss:.4f}")

if __name__ == '__main__':
    main()
