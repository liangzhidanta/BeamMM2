# data_loader_decoder.py

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import os
from PIL import Image

class CustomDataset_decoder(Dataset):
    def __init__(self, data_csv_paths, transform=None, modal='mmwave_gps', input_length=8, output_length=3):
        self.data_csv_paths = data_csv_paths
        self.transform = transform
        self.modal = modal
        self.input_length = input_length
        self.output_length = output_length

        assert modal in ['gps', 'mmwave', 'mmwave_gps']
        
        base_paths = [os.path.dirname(data_csv_path) for data_csv_path in self.data_csv_paths]
        self.seq_base_paths = []  # 每个序列的基础路径

        self.features_column = {  # 特征名: 数据集中的列名
            'rgbs': 'unit1_rgb',
            'u1_loc': 'unit1_loc',
            'u2_loc': 'unit2_loc',
            'mmwave': 'unit1_pwr_60ghz',
        }
        
        self.window_samples = []  # 存储所有滑动窗口的样本信息
        self.scenario_list = []   # 存储每个序列对应的场景名称

        # 遍历所有CSV文件，生成滑动窗口样本
        for seq_idx, data_csv_path in enumerate(self.data_csv_paths):
            scenario = os.path.basename(os.path.dirname(data_csv_path))  # e.g., 'scenario1'
            # 记录序列对应的场景
            # 每个序列可能有多个滑动窗口样本
            data_csv = pd.read_csv(data_csv_path)
            seq_ids = data_csv['seq_index'].unique()
            for seq_id in seq_ids:
                seq_data = data_csv[data_csv['seq_index'] == seq_id]
                total_length = len(seq_data)
                # required_length = self.input_length + self.output_length  # 需要获取的总数据长度
                required_length = self.input_length  # 需要获取的总数据长度
                if total_length < required_length:
                    # 跳过不足长度的序列
                    continue
                # 计算所有可能的滑动窗口起始索引
                for start_idx in range(total_length - required_length + 1):
                    self.window_samples.append((seq_idx, seq_id, start_idx))
                    self.scenario_list.append(scenario)

        # # 遍历所有CSV文件，生成滑动窗口样本
        # for seq_idx, data_csv_path in enumerate(self.data_csv_paths):
        #     scenario = os.path.basename(os.path.dirname(data_csv_path))  # e.g., 'scenario1'
        #     # 记录序列对应的场景
        #     # 每个序列可能有多个滑动窗口样本
        #     data_csv = pd.read_csv(data_csv_path)
        #     seq_ids = data_csv['seq_index'].unique()
        #     for seq_id in seq_ids:
        #         seq_data = data_csv[data_csv['seq_index'] == seq_id]
        #         total_length = len(seq_data)
        #         # required_length = self.input_length + self.output_length  # 需要获取的总数据长度
        #         required_length = self.input_length  # 需要获取的总数据长度
        #         if total_length < required_length:
        #             # 跳过不足长度的序列
        #             continue
        #         # 计算所有可能的滑动窗口起始索引，间隔为2
        #         for start_idx in range(0, total_length - required_length + 1, 2):
        #             self.window_samples.append((seq_idx, seq_id, start_idx))
        #             self.scenario_list.append(scenario)

    def __len__(self):
        return len(self.window_samples)
    
    def __getitem__(self, idx):
        seq_idx, seq_id, start_idx = self.window_samples[idx]
        data_csv_path = self.data_csv_paths[seq_idx]
        base_path = os.path.dirname(data_csv_path)
        data_csv = pd.read_csv(data_csv_path)
        seq_data = data_csv[data_csv['seq_index'] == seq_id]

        # 获取特定滑动窗口的数据
        window_data = {}
        if self.modal == 'gps':
            to_get = ['rgbs', 'u1_loc', 'u2_loc']
        elif self.modal == 'mmwave':
            to_get = ['rgbs', 'mmwave']
        elif self.modal == 'mmwave_gps':
            to_get = ['rgbs', 'u1_loc', 'u2_loc', 'mmwave']
        
        for feature_name in to_get:
            window_data[feature_name] = seq_data[self.features_column[feature_name]].iloc[start_idx:start_idx + self.input_length + self.output_length].tolist()
        
        # 处理输入部分：前 `input_length` 个数据
        video = []
        for i in range(self.input_length):
            image_path = os.path.join(base_path, window_data['rgbs'][i])
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            video.append(image)
        video = torch.stack(video)  # [input_length, C, H, W]
        
        # 处理 GPS 数据
        if self.modal in ['gps', 'mmwave_gps']:
            gps = []
            for u1_loc_path, u2_loc_path in zip(
                window_data['u1_loc'][:self.input_length],
                window_data['u2_loc'][:self.input_length]
            ):
                with open(os.path.join(base_path, u1_loc_path), 'r') as f:
                    lat1, lon1 = map(float, f.read().strip().split())
                
                with open(os.path.join(base_path, u2_loc_path), 'r') as f:
                    lat2, lon2 = map(float, f.read().strip().split())
                    gps_single = torch.tensor([lat2 - lat1, lon2 - lon1], dtype=torch.float32)
                    gps.append(gps_single)
            gps = torch.stack(gps)  # [input_length, 2]
        else:
            gps = None
        
        # 处理 mmWave 数据
        if self.modal in ['mmwave', 'mmwave_gps']:
            mmwave = []
            for mmwave_path in window_data['mmwave'][:self.input_length]:
                with open(os.path.join(base_path, mmwave_path), 'r') as f:
                    mmwave_single = torch.tensor(list(map(float, f.read().strip().split())), dtype=torch.float32)
                    mmwave.append(mmwave_single)
            mmwave = torch.stack(mmwave)  # [input_length, D]
        else:
            mmwave = None
        
        # 处理目标数据：目标是输入的后 `output_len` 个数据
        target_mmwave = []
        for mmwave_path in window_data['mmwave'][self.input_length - self.output_length:self.input_length]:
            with open(os.path.join(base_path, mmwave_path), 'r') as f:
                mmwave_single = torch.tensor(list(map(float, f.read().strip().split())), dtype=torch.float32)
                target_mmwave.append(mmwave_single)
        target_mmwave = torch.stack(target_mmwave)  # [output_length, D]
        
        # 根据模态返回输入和目标
        if self.modal == 'gps':
            return (video, gps), target_mmwave
        elif self.modal == 'mmwave':
            return (video, mmwave), target_mmwave
        elif self.modal == 'mmwave_gps':
            return (video, gps, mmwave), target_mmwave

def collate_fn(batch):
    inputs, targets = zip(*batch)
    if len(inputs[0]) == 2:
        # modal: 'mmwave'
        videos, mmwaves = zip(*inputs)
        videos = pad_sequence(videos, batch_first=True)  # [B, input_length, C, H, W]
        mmwaves = pad_sequence(mmwaves, batch_first=True)  # [B, input_length, D]
        targets = pad_sequence(targets, batch_first=True)  # [B, output_length, D]
        res = {
            'video': videos,
            'mmwave': mmwaves,
            'target_mmwave': targets,
        }
    elif len(inputs[0]) == 3:
        # modal: 'mmwave_gps'
        videos, gpses, mmwaves = zip(*inputs)
        videos = pad_sequence(videos, batch_first=True)  # [B, input_length, C, H, W]
        gpses = pad_sequence(gpses, batch_first=True)    # [B, input_length, 2]
        mmwaves = pad_sequence(mmwaves, batch_first=True)  # [B, input_length, D]
        targets = pad_sequence(targets, batch_first=True)  # [B, output_length, D]
        res = {
            'video': videos,
            'gps': gpses,
            'mmwave': mmwaves,
            'target_mmwave': targets,
        }
    else:
        raise ValueError("Unsupported batch format.")
    return res