# data_loader.py
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import os
from PIL import Image

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, data_csv_paths, transform=None, modal='mmwave', input_length=8, output_length=0):
        self.data_csv_paths = data_csv_paths
        self.transform = transform
        self.input_length = input_length
        self.output_length = output_length

        assert modal in ['gps', 'mmwave', 'mmwave_gps']
        self.modal = modal

        self.base_paths = [os.path.dirname(data_csv_path) for data_csv_path in data_csv_paths]

        self.features_column = {  # 特征名: 数据集中的列名
            'rgbs': 'unit1_rgb',
            'u1_loc': 'unit1_loc',
            'u2_loc': 'unit2_loc',
            'mmwave': 'unit1_pwr_60ghz',
        }

        self.window_samples = []  # 存储所有滑动窗口的样本信息
        self.scenario_list = []   # 存储每个序列对应的场景名称

        for scenario_idx, data_csv_path in enumerate(self.data_csv_paths):
            scenario = os.path.basename(os.path.dirname(data_csv_path))  #  e.g., 'scenario1'
            # 记录序列对应的场景
            # 每个序列可能有多个滑动窗口样本
            data_csv = pd.read_csv(data_csv_path)
            seq_ids = data_csv['seq_index'].unique()
            for seq_id in seq_ids:
                seq_data = data_csv[data_csv['seq_index'] == seq_id]
                total_length = len(seq_data)
                required_length = self.input_length + self.output_length
                if total_length < required_length:
                    # 跳过不足长度的序列
                    continue
                # 计算所有可能的滑动窗口起始索引
                for start_idx in range(total_length - required_length + 1):
                    self.window_samples.append((scenario_idx, seq_id, start_idx))  # 序列索引，序列id，滑动窗口起始索引
                    self.scenario_list.append(scenario)
                    # 这里假设每个滑动窗口样本对应一个场景
                    # 如果需要更详细的场景信息，可以调整存储方式

    def __len__(self):
        return len(self.window_samples)
    
    def __getitem__(self, idx):
        senario_idx, seq_id, start_idx = self.window_samples[idx]
        base_path = self.base_paths[senario_idx]

        data_csv = pd.read_csv(self.data_csv_paths[senario_idx])
        seq_data = data_csv[data_csv['seq_index'] == seq_id]

        # 获取滑动窗口数据
        window_data = {}

        if self.modal == 'gps':
            to_get = ['rgbs', 'u1_loc', 'u2_loc']
        elif self.modal == 'mmwave':
            to_get = ['rgbs', 'mmwave']
        elif self.modal == 'mmwave_gps':
            to_get = ['rgbs', 'u1_loc', 'u2_loc', 'mmwave']
        
        for feature_name in to_get:
            window_data[feature_name] = seq_data[self.features_column[feature_name]].iloc[start_idx: start_idx + self.input_length + self.output_length].tolist()
        # rgbs, u1_loc, u2_loc, mmwave = seq_data['rgbs'], seq_data['u1_loc'], seq_data['u2_loc'], seq_data['mmwave']
        
        video = []

        for i in range(self.input_length):
            image = Image.open(os.path.join(base_path, window_data['rgbs'][i])).convert('RGB') 
            if self.transform:
                image = self.transform(image)
            video.append(image)
            
        video = torch.stack(video)  # [input_length, C, H, W]

        if self.modal in ['gps', 'mmwave_gps']:
            gps = []
            for u1_loc_path, u2_loc_path in zip(
                window_data['u1_loc'][:self.input_length], window_data['u2_loc'][:self.input_length]
            ):
                with open(os.path.join(base_path, u1_loc_path), 'r') as f:
                    lat1, lon1 = map(float, f.read().strip().split())
                
                with open(os.path.join(base_path, u2_loc_path), 'r') as f:
                    lat2, lon2 = map(float, f.read().strip().split())
                    gps_single = torch.tensor([lat2-lat1, lon2-lon1], dtype=torch.float32)
                    gps.append(gps_single)
            gps = torch.stack(gps)
        else:
            gps = None

            
        if self.modal in ['mmwave', 'mmwave_gps']:
            mmwave = []
            for mmwave_path in window_data['mmwave'][:self.input_length]:
                with open(os.path.join(base_path, mmwave_path), 'r') as f:
                    mmwave_single = torch.tensor(list(map(float, f.read().strip().split())), dtype=torch.float32)
                    mmwave.append(mmwave_single)
            mmwave = torch.stack(mmwave)
        else:
            mmwave = None

        # 根据模态返回输入和目标
        if self.modal == 'gps':
            return {
                'video': video,
                'gps': gps,
            }
        elif self.modal == 'mmwave':
            return {
                'video': video,
                'mmwave': mmwave,
            }
        elif self.modal == 'mmwave_gps':
            return {
                'video': video,
                'gps': gps,
                'mmwave': mmwave,
            }


def collate_fn(batch):
    if len(batch[0]) == 2:
        batch_size = len(batch)
        max_video_length = max([video.shape[0] for video, _ in batch])
        max_mmwave_length = max([mmwave.shape[0] for _, mmwave in batch])
        # mask_video = torch.zeros(batch_size, max_video_length)
        # mask_mmwave = torch.zeros(batch_size, max_mmwave_length)
        fea_video = []
        fea_mmwave = []
        for i in range(batch_size):
            video, mmwave = batch[i]
            video_length = video.shape[0]
            mmwave_length = mmwave.shape[0]
            fea_video.append(video)
            fea_mmwave.append(mmwave)
            # mask_video[i, :video_length] = 1
            # mask_mmwave[i, :mmwave_length] = 1
        
        res = {
            'video': pad_sequence(fea_video, batch_first=True),
            'mmwave': pad_sequence(fea_mmwave, batch_first=True),
            # 'mask_video': mask_video,
            # 'mask_mmwave': mask_mmwave,
        }
    elif len(batch[0]) == 3:
        batch_size = len(batch)
        max_video_length = max([video.shape[0] for video, _, _ in batch])
        max_mmwave_length = max([mmwave.shape[0] for _, mmwave, _ in batch])
        max_gps_length = max([gps.shape[0] for _, gps, _ in batch])

        fea_video = []
        fea_gps = []
        fea_mmwave = []
        for i in range(batch_size):
            video, gps, mmwave = batch[i]
            video_length = video.shape[0]
            gps_length = gps.shape[0]
            mmwave_length = mmwave.shape[0]
            fea_video.append(video)
            fea_gps.append(gps)
            fea_mmwave.append(mmwave)
        
        res = {
            'video': pad_sequence(fea_video, batch_first=True),
            'gps': pad_sequence(fea_gps, batch_first=True),
            'mmwave': pad_sequence(fea_mmwave, batch_first=True),
        }

    return res
    

if __name__ == '__main__':
    pass