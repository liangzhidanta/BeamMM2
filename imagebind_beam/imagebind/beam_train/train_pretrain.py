import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # 导入 tqdm
import random
import numpy as np
import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录的父目录，即 /home/wzj/wyh/ImageBind_beam
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 将其添加到 sys.path
sys.path.insert(0, parent_parent_dir)
from imagebind.models.imagebind_model import ModalityType, ImageBindModel

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # 如果使用的是 cudnn，以下设置可以进一步提高可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 选择一个固定的随机种子

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义转换
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 根据需要调整
                         std=[0.229, 0.224, 0.225]),
])

# 实例化数据集
train_dataset = CustomDataset(
    image_dir="/new_disk/yy/DeepSensePre/Merged_data/camera_data",
    gps_dir="/new_disk/yy/DeepSensePre/Merged_data/GPS_data",
    mmwave_dir="/new_disk/yy/DeepSensePre/Merged_data/mmWave_data",
    transform=train_transforms
)

val_dataset = CustomDataset(
    image_dir="/new_disk/yy/DeepSensePre/Merged_data/camera_data",
    gps_dir="/new_disk/yy/DeepSensePre/Merged_data/GPS_data",
    mmwave_dir="/new_disk/yy/DeepSensePre/Merged_data/mmWave_data",
    transform=train_transforms
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 初始化模型（使用默认参数）
model = ImageBindModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def info_nce_loss(anchor, positive, temperature=0.07):
    """
    计算 InfoNCE 损失，anchor 和 positive 的形状均为 [B, D]
    """
    batch_size = anchor.shape[0]
    labels = torch.arange(batch_size).to(anchor.device)
    similarity_matrix = nn.functional.cosine_similarity(anchor.unsqueeze(1), positive.unsqueeze(0), dim=2)
    similarity_matrix = similarity_matrix / temperature
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
    return loss

best_loss = float("inf")
patience = 3
wait = 0
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # 创建训练进度条
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for batch_idx, (images, gps, mmwave) in enumerate(train_progress):
        images, gps, mmwave = images.to(device), gps.to(device), mmwave.to(device)
        
        optimizer.zero_grad()
        inputs = {
            ModalityType.VISION: images,
            ModalityType.GPS: gps,
            ModalityType.MMWAVE: mmwave
        }
        outputs = model(inputs)
        
        # 提取各模态特征
        vision_features = outputs[ModalityType.VISION]      # [B, D]
        gps_features = outputs[ModalityType.GPS]            # [B, D]
        mmwave_features = outputs[ModalityType.MMWAVE]      # [B, D]
        
        # 计算 (image, GPS) 的 InfoNCE 损失
        loss_image_gps = info_nce_loss(vision_features, gps_features)
        
        # 计算 (image, mmWave) 的 InfoNCE 损失
        loss_image_mmwave = info_nce_loss(vision_features, mmwave_features)
        
        # 综合损失（可以根据需要调整权重）
        loss = loss_image_gps + loss_image_mmwave
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # 更新进度条的描述
        train_progress.set_postfix({'Batch Loss': loss.item()})
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}')
    
    # 验证阶段同理
    model.eval()
    val_loss = 0.0
    
    # 创建验证进度条
    val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch_idx, (images, gps, mmwave) in enumerate(val_progress):
            images, gps, mmwave = images.to(device), gps.to(device), mmwave.to(device)
            
            inputs = {
                ModalityType.VISION: images,
                ModalityType.GPS: gps,
                ModalityType.MMWAVE: mmwave
            }
            outputs = model(inputs)
            
            # 提取各模态特征
            vision_features = outputs[ModalityType.VISION]
            gps_features = outputs[ModalityType.GPS]
            mmwave_features = outputs[ModalityType.MMWAVE]
            
            # 计算与图像模态的 InfoNCE 损失
            loss_image_gps = info_nce_loss(vision_features, gps_features)
            loss_image_mmwave = info_nce_loss(vision_features, mmwave_features)
            
            # 综合损失
            loss = loss_image_gps + loss_image_mmwave
            val_loss += loss.item()
            
            # 更新进度条的描述
            val_progress.set_postfix({'Val Batch Loss': loss.item()})
    
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
    
    # 早停和保存检查点
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

print('Finished Training')

# 加载最佳模型进行最终测试
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
total_loss = 0.0

# 创建测试进度条
test_progress = tqdm(val_loader, desc="Final Test [Val]", leave=False)
with torch.no_grad():
    for batch_idx, (images, gps, mmwave) in enumerate(test_progress):
        images, gps, mmwave = images.to(device), gps.to(device), mmwave.to(device)
        inputs = {
            ModalityType.VISION: images,
            ModalityType.GPS: gps,
            ModalityType.MMWAVE: mmwave
        }
        outputs = model(inputs)
        
        # 提取各模态特征
        vision_features = outputs[ModalityType.VISION]
        gps_features = outputs[ModalityType.GPS]
        mmwave_features = outputs[ModalityType.MMWAVE]
        
        # 计算与图像模态的 InfoNCE 损失
        loss_image_gps = info_nce_loss(vision_features, gps_features)
        loss_image_mmwave = info_nce_loss(vision_features, mmwave_features)
        
        # 综合损失
        loss = loss_image_gps + loss_image_mmwave
        total_loss += loss.item()
        
        # 更新进度条的描述
        test_progress.set_postfix({'Test Batch Loss': loss.item()})

print(f'Final Reconstruction Loss: {total_loss/len(val_loader):.4f}')
