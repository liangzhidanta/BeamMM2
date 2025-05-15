# test_MMWaveEncoderDecoderModel.py

import torch
from data_loader_decoder import CustomDataset_decoder, collate_fn
from torchvision import transforms
import glob
import os
from collections import defaultdict, OrderedDict
import math
import random
from tqdm import tqdm  # 引入 tqdm 库
import argparse  # 引入 argparse 模块
import sys
sys.path.append('./imagebind_beam')
from imagebind.models.imagebind_model import ModalityType, ImageBindModel
from model import BeamVisionEncoderDecoderModel
from decoder import MMWaveDecoder
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

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
        if var.item() == 0:
            return torch.tensor(float('inf')).to(output.device)
        return mse / var

# prediction和target的topk重合程度指标
def accuracy_at_k(predictions, targets, k=1):
    """
    predictions: shape = [batch_size, seq_length, num_classes]
    targets: shape = [batch_size, seq_length, num_classes]
    k: 取前 k 的索引
    """
    batch_size = predictions.size(0)
    seq_length = predictions.size(1)

    # 取预测和目标的前k个索引
    _, topk_indices_pred = predictions.topk(k, dim=-1, largest=True, sorted=True)
    _, topk_indices_true = targets.topk(k, dim=-1, largest=True, sorted=True)

    # 统计总的命中数
    total_hit = 0
    for i in range(batch_size):
        for j in range(seq_length):
            pred_set = set(topk_indices_pred[i, j].tolist())
            true_set = set(topk_indices_true[i, j].tolist())
            overlap = pred_set.intersection(true_set)
            total_hit += len(overlap)

    # 除以总数得到比例
    total = batch_size * seq_length * k  # 总的预测索引总数
    precision = total_hit / total
    return precision


# 定义测试评估函数
def test_evaluate(model, data_loader, mse_criterion, nmse_criterion, device, modal='mmwave_gps'):
    model.eval()
    epoch_loss_mse = 0.0
    epoch_loss_nmse = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Testing', leave=False)):
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
            # tgt_seq = torch.zeros(target_mmwave.size(0), target_mmwave.size(1), dtype=torch.long, device=device)  # [B, output_length]
            tgt_seq = torch.zeros(target_mmwave.size(0), 3, dtype=torch.long, device=device)  # [B, output_length]

            # 前向传播
            output = model(inputs, tgt_seq)  # [B, output_length, D]

            # 计算 MSE 和 NMSE 损失
            loss_mse = mse_criterion(output, target_mmwave)
            loss_nmse = nmse_criterion(output, target_mmwave)

            epoch_loss_mse += loss_mse.item()
            epoch_loss_nmse += loss_nmse.item()

            # 收集预测和真实值
            all_predictions.append(output)
            all_targets.append(target_mmwave)

    avg_loss_mse = epoch_loss_mse / len(data_loader)
    avg_loss_nmse = epoch_loss_nmse / len(data_loader)
    
    # 拼接所有批次的预测值和目标值
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    accuracy1 =accuracy_at_k(all_predictions, all_targets, k=1)
    accuracy5 = accuracy_at_k(all_predictions, all_targets, k=5)

    return avg_loss_mse, avg_loss_nmse, accuracy1, accuracy5

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
    parser = argparse.ArgumentParser(description="Testing MMWaveEncoderDecoderModel")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--model_path', type=str, default="./checkpoints", help='模型权重的路径')
    parser.add_argument('--dataset_start_idx', type=int, default=1, help='数据集的起始索引')
    parser.add_argument('--dataset_end_idx', type=int, default=9, help='数据集的结束索引')
    args = parser.parse_args()

    # 定义转换，与预训练时保持一致
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 数据集路径
    # dataset_path = [f'/new_disk/yy/DeepSensePre/Data_raw/scenario{i}/' for i in range(1, 2)]  # scenario1 ~ scenario1
    dataset_path = [f'/new_disk/yy/DeepSensePre/Data_raw/scenario{i}/' for i in range(args.dataset_start_idx, args.dataset_end_idx)]  # scenario1 ~ scenario8
    data_csv_paths = []
    for path in dataset_path:
        data_csv_paths.extend(glob.glob(os.path.join(path, '*.csv')))

    print(f"Found {len(data_csv_paths)} CSV files for testing.")

    # 初始化 CustomDataset_decoder
    modal = 'mmwave_gps'
    # modal = 'mmwave'
    input_length = 8
    output_length = 3
    dataset = CustomDataset_decoder(
        data_csv_paths,
        transform=default_transform,
        modal=modal,
        input_length=input_length,
        output_length=output_length
    )

    # # 按场景划分训练集、验证集和测试集
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

    # 创建 Subset 数据集
    # test_dataset = Subset(dataset, test_indices)

    # 创建 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 选择模型、损失函数等并进行测试评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    # 模型加载
    # 初始化编码器
    encoder = ImageBindModel(
        video_frames=input_length,  # 与数据加载时的 input_length 一致
        # 其他参数应与预训练时保持一致
    ).to(device)

    # 初始化集成模型，提供各模态的特征维度
    vision_dim = 768  # 假设 VISION 特征维度为 768
    gps_dim = 768     # 假设 GPS 特征维度为 768
    mmwave_dim = 768  # 确保 mmwave_dim 设置为 768

    # 初始化集成模型
    model = BeamVisionEncoderDecoderModel(
        encoder=encoder, 
        decoder=decoder, 
        vision_dim=vision_dim, 
        beam_dim=mmwave_dim,
        input_length=input_length,
        output_length=output_length
    ).to(device)

    # 加载预训练的编码器权重
    model_save_path = args.model_path  # 使用命令行参数提供的路径
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model weights not found at {model_save_path}")

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
    model.load_state_dict(new_state_dict)
    print("Encoder weights loaded successfully.")

    # 损失函数
    mse_criterion = nn.MSELoss()
    nmse_criterion = NMSELoss()

    # 评估模型
    avg_loss_mse, avg_loss_nmse, accuracy1, accuracy5 = test_evaluate(
        model, test_loader, mse_criterion, nmse_criterion, device, modal=modal
    )

    # 输出测试结果
    print(f"Test Loss (MSE): {avg_loss_mse:.4f}")
    print(f"Test Loss (NMSE): {avg_loss_nmse:.4f}")
    print(f"Accuracy@1: {accuracy1:.4f}")
    print(f"Accuracy@5: {accuracy5:.4f}")

if __name__ == '__main__':
    main()
