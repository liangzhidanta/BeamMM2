import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, image_dir, gps_dir, mmwave_dir, transform=None):
        self.image_dir = image_dir
        self.gps_dir = gps_dir
        self.mmwave_dir = mmwave_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.gps_files = sorted(os.listdir(gps_dir))
        self.mmwave_files = sorted(os.listdir(mmwave_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        gps_path = os.path.join(self.gps_dir, self.gps_files[idx])
        mmwave_path = os.path.join(self.mmwave_dir, self.mmwave_files[idx])

        # 加载图像
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # 调试：确认图像已转换为Tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Image at index {idx} is not a torch.Tensor after transformation.")

        # 加载GPS数据
        try:
            with open(gps_path, 'r') as f:
                lat, lon = map(float, f.read().strip().split())
                gps = torch.tensor([lat, lon], dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Error loading GPS data at index {idx}: {e}")

        # 加载mmWave数据
        try:
            with open(mmwave_path, 'r') as f:
                mmwave = torch.tensor([float(v) for v in f.read().strip().split()], dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Error loading mmWave data at index {idx}: {e}")

        # 调试：确认GPS和mmWave也是Tensor
        if not isinstance(gps, torch.Tensor):
            raise TypeError(f"GPS data at index {idx} is not a torch.Tensor.")
        if not isinstance(mmwave, torch.Tensor):
            raise TypeError(f"mmWave data at index {idx} is not a torch.Tensor.")

        return image, gps, mmwave

# 使用示例
if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711),),
    ])

    dataset = CustomDataset(image_dir='/new_disk/yy/DeepSensePre/Merged_data/camera_data', 
                            gps_dir='/new_disk/yy/DeepSensePre/Merged_data/GPS_data', 
                            mmwave_dir='/new_disk/yy/DeepSensePre/Merged_data/mmWave_data', 
                            transform=transform)
    # "/new_disk/yy/DeepSensePre/Merged_data/camera_data", "/new_disk/yy/DeepSensePre/Merged_data/GPS_data", "/new_disk/yy/DeepSensePre/Merged_data/mmWave_data"
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, gps, mmwave in dataloader:
        print(images.shape, gps.shape, mmwave.shape)
        break
