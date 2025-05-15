# pretrain.py
from data_loader import CustomDataset, collate_fn

import glob
import os

# 用于训练的 GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 程序中GPU的编号变为0，1.(0表示系统第2张GPU)

import random
import time
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import amp
use_amp = False

import sys
sys.path.append('./imagebind_beam')
from imagebind.models.imagebind_model import ModalityType, ImageBindModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model_save_path = './checkpoints/best_model.pth'
optimizer_save_path = './checkpoints/best_optimizer.pth'

batch_size = 8

# 定义转换
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 根据需要调整
                         std=[0.229, 0.224, 0.225]),
])

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

cur_time = time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))

def info_nce_loss(anchor, positive, temperature=0.07):
    """
    计算 InfoNCE 损失，anchor 和 positive 的形状均为 [B, S, D]
    Args:
        anchor (torch.Tensor): 锚点特征，形状为 [B, S, D]
        positive (torch.Tensor): 正样本特征，形状为 [B, S, D]
        temperature (float): 温度参数，用于调整相似度的尺度
    Returns:
        loss (torch.Tensor): InfoNCE 损失值
    """
    # L2 归一化
    anchor = F.normalize(anchor, dim=2)  # [B, S, D]
    positive = F.normalize(positive, dim=2)  # [B, S, D]

    B, S, D = anchor.shape

    # 将 Batch Size 和 Sequence Length 合并
    anchor = anchor.view(B * S, D)  # [B*S, D]
    positive = positive.view(B * S, D)  # [B*S, D]

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(anchor, positive.T)  # [B*S, B*S]
    similarity_matrix = similarity_matrix / temperature  # 缩放相似度

    # 构造标签
    labels = torch.arange(B * S).to(anchor.device)  # [B*S]

    # 计算 InfoNCE 损失
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def save_best(model, optimizer, model_save_path=model_save_path, optimizer_save_path=optimizer_save_path):
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)


dataset_path = [f'/new_disk/yy/DeepSensePre/Data_raw/scenario{i}/' for i in [1]]  # scenario1
# dataset_path = [f'/data/wzj/Datasets/Scenario{i}/' for i in range(1, 10)]  # scenario1 ~ scenario9

data_csv_paths = []
for path in dataset_path:
    data_csv_paths.extend(glob.glob(path + '*.csv'))
print(data_csv_paths)


def mmwave_pretrain():
    dataset = CustomDataset(data_csv_paths, transform=default_transform, modal='mmwave')

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ImageBindModel(
        video_frames=8,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')
    wait = 0 
    patience = 30
    num_epochs = 50

    # train
    for epoch in range(num_epochs):
        model.train()

        epoch_loss = []
        for batch_idx, batch in enumerate(train_loader):
            video, mmwave = batch['video'].to(device), batch['mmwave'].to(device)

            optimizer.zero_grad()
            inputs = {
                ModalityType.VISION: video,
                ModalityType.MMWAVE: mmwave,
            }
            outputs = model(inputs)

            vision_features = outputs[ModalityType.VISION]
            mmwave_features = outputs[ModalityType.MMWAVE]

            # print('vision_features', vision_features.shape)
            # print('mmwave_features', mmwave_features.shape)

            loss = info_nce_loss(vision_features, mmwave_features)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {np.mean(epoch_loss)}')
    

def gps_pretrain():
    dataset = CustomDataset(data_csv_paths, transform=default_transform, modal='gps')

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ImageBindModel(
        video_frames=8,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')
    wait = 0 
    patience = 5
    num_epochs = 50

    # train
    for epoch in range(num_epochs):
        model.train()

        epoch_loss = []
        for batch_idx, batch in enumerate(train_loader):
            video, gps = batch['video'].to(device), batch['mmwave'].to(device)
            print('gps shape', gps.shape)

            optimizer.zero_grad()
            inputs = {
                ModalityType.VISION: video,
                ModalityType.GPS: gps,
            }
            outputs = model(inputs)

            vision_features = outputs[ModalityType.VISION]
            gps_features = outputs[ModalityType.GPS]

            # print('vision_features', vision_features.shape)
            # print('mmwave_features', mmwave_features.shape)

            loss = info_nce_loss(vision_features, gps_features)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {np.mean(epoch_loss)}')


def mmwave_gps_joint_pretrain(checkpoint_path=None, optimizer_path=None, start_epoch=0, best_loss=None):
    dataset = CustomDataset(data_csv_paths, transform=default_transform, modal='mmwave_gps')
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=64)
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model = ImageBindModel(
            video_frames=8,
        ).to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model = ImageBindModel(
            video_frames=8,
        ).to(device)
        model = nn.DataParallel(model)
    
    if optimizer_path is not None and os.path.exists(optimizer_path):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 500
    wait = 0
    patience = 10
    best_loss = float('inf') if best_loss is None else best_loss

    scaler = amp.GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        # train
        model.train()
        epoch_loss = []

        for batch_idx, batch in enumerate(train_loader):
            video, mmwave, gps = batch['video'].to(device), batch['mmwave'].to(device), batch['gps'].to(device)

            optimizer.zero_grad()
            inputs = {
                ModalityType.VISION: video,
                ModalityType.MMWAVE: mmwave,
                ModalityType.GPS: gps,
            }
            # print inputs shape
            # print('I video', video.shape)
            # print('I mmwave', mmwave.shape)
            # print('I gps', gps.shape)

            with amp.autocast(enabled=use_amp):
                outputs = model(inputs)

                vision_features = outputs[ModalityType.VISION]
                mmwave_features = outputs[ModalityType.MMWAVE]
                gps_features = outputs[ModalityType.GPS]

                # print shape of features
                # print('O vision_features', vision_features.shape)
                # print('O mmwave_features', mmwave_features.shape)
                # print('O gps_features', gps_features.shape)

                loss_vision_mmwave = info_nce_loss(vision_features, mmwave_features)
                loss_vision_gps = info_nce_loss(vision_features, gps_features)
                loss = loss_vision_mmwave + loss_vision_gps

            epoch_loss.append([loss_vision_mmwave.item(), loss_vision_gps.item(), loss.item()])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

        print(f'Epoch {epoch}, Training Loss: {np.mean([x[2] for x in epoch_loss])}')
        print(f'\tTraining Loss Vision-MMWave: {np.mean([x[0] for x in epoch_loss])}')
        print(f'\tTraing Loss Vision-GPS: {np.mean([x[1] for x in epoch_loss])}')

        # valiadation
        model.eval()
        val_loss = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                video, mmwave, gps = batch['video'].to(device), batch['mmwave'].to(device), batch['gps'].to(device)

                inputs = {
                    ModalityType.VISION: video,
                    ModalityType.MMWAVE: mmwave,
                    ModalityType.GPS: gps,
                }
                # with amp.autocast(enabled=use_amp):
                outputs = model(inputs)

                vision_features = outputs[ModalityType.VISION]
                mmwave_features = outputs[ModalityType.MMWAVE]
                gps_features = outputs[ModalityType.GPS]

                loss_vision_mmwave = info_nce_loss(vision_features, mmwave_features)
                loss_vision_gps = info_nce_loss(vision_features, gps_features)
                loss = loss_vision_mmwave + loss_vision_gps
                val_loss.append([loss_vision_mmwave.item(), loss_vision_gps.item(), loss.item()])

        print(f'Epoch {epoch}, Loss: {np.mean([x[2] for x in val_loss])}')
        print(f'\tLoss Vision-MMWave: {np.mean([x[0] for x in val_loss])}')
        print(f'\tLoss Vision-GPS: {np.mean([x[1] for x in val_loss])}')
        
        epoch_val_loss_mean = np.mean([x[2] for x in val_loss])
        if epoch_val_loss_mean < best_loss:
            
            cur_model_save_path = checkpoint_path if checkpoint_path else f'checkpoints/mmwave_gps_joint-{cur_time}.pth'
            cur_optimizer_save_path = optimizer_path if optimizer_path else f'checkpoints/mmwave_gps_joint-{cur_time}_optimizer.pth'
            save_best(
                model, optimizer,
                model_save_path=cur_model_save_path,
                optimizer_save_path=cur_optimizer_save_path
            )

            best_loss = epoch_val_loss_mean
            print(f'Best model saved at epoch {epoch}')
            wait = 0
        else:
            wait += 1
            if wait > patience:
                print(f'Early stopping at epoch {epoch}')
                exit(0)

        end_time = time.time()
        print(f'\tEpoch time: {end_time - start_time}')
        

if __name__ == '__main__':
    # mmwave_pretrain()
    # gps_pretrain()
    mmwave_gps_joint_pretrain(
        # checkpoint_path='checkpoints/mmwave_gps_joint-01-14-13:35:40.pth',
        # optimizer_path='checkpoints/mmwave_gps_joint-01-14-13:35:40_optimizer.pth',
        # start_epoch=47,
        # best_loss=0.34205562517135624
    )