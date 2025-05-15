# encoder_inference.py
import torch
from data_loader_decoder import CustomDataset_decoder, collate_fn
from torchvision import transforms
import glob
import os
import sys
sys.path.append('./imagebind_beam')
print(sys.path)
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
from collections import OrderedDict

# 1. 数据加载部分

# 定义转换，与预训练时保持一致
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 与预训练时一致
                         std=[0.229, 0.224, 0.225]),
])

# 定义数据集路径
dataset_path = [f'/new_disk/yy/DeepSensePre/Data_raw/scenario{i}/' for i in range(1, 2)]  # scenario1 ~ scenario9
data_csv_paths = []
for path in dataset_path:
    data_csv_paths.extend(glob.glob(os.path.join(path, '*.csv')))

print(f"Found {len(data_csv_paths)} CSV files for inference.")

# 初始化 CustomDataset
# modal = 'mmwave_gps'  # 选择模态
modal = 'mmwave'  # 选择模态
dataset = CustomDataset_decoder(data_csv_paths, transform=default_transform, modal=modal)

# 创建 DataLoader
batch_size = 4
inference_loader = torch.utils.data.DataLoader(
    dataset,
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

# 初始化模型
model = ImageBindModel(
    video_frames=8,  # 确保与预训练时一致
    # 根据需要添加其他参数
).to(device)

# 定义模型权重路径
# model_save_path = '/home/wzj/BeamMM/checkpoints/mmwave_gps_joint-01-10-21:57:09.pth'  # 根据实际路径修改
# model_save_path = '/home/wzj/BeamMM/checkpoints/mmwave_gps_joint-01-12-182501.pth'
model_save_path = '/home/wzj/BeamMM/checkpoints/mmwave_gps_joint-01-15-152626.pth'

# 加载保存的state_dict
state_dict = torch.load(model_save_path, map_location=device)
print(f"Loaded state_dict from {model_save_path}")

# 处理多GPU保存的模型（如果适用）
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('module.'):
        # new_state_dict[k[7:]] = v
        # 去除键中的"module."前缀
        name = k.replace("module.", "")
        # 将新的键和值添加到新的字典中
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

# 加载权重到模型
model.load_state_dict(new_state_dict)
print("Model weights loaded successfully.")

# 设置模型为评估模式
model.eval()
print("Model set to evaluation mode.")

# 3. 推理部分

def run_inference(model, data_loader, device, modal):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
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

            # 前向传播
            outputs = model(inputs)

            # 处理输出
            vision_features = outputs.get(ModalityType.VISION)
            gps_features = outputs.get(ModalityType.GPS)
            mmwave_features = outputs.get(ModalityType.MMWAVE)

            # 示例：打印特征向量的形状
            print(f"Batch {batch_idx}:")
            if vision_features is not None:
                print(f"  Vision Features Shape: {vision_features.shape}")
            if gps_features is not None:
                print(f"  GPS Features Shape: {gps_features.shape}")
            if mmwave_features is not None:
                print(f"  mmWave Features Shape: {mmwave_features.shape}")

            # 根据需求，进一步处理特征向量
            # 例如，保存特征、进行分类等

            # 示例：只处理第一批数据
            break  # 注释掉以处理整个数据集

# 运行推理
run_inference(model, inference_loader, device, modal)