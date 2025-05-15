# model.py

import torch
import torch.nn as nn
import sys
sys.path.append('./imagebind_beam')
from imagebind.models.imagebind_model import ModalityType, ImageBindModel
from decoder import MMWaveDecoder

class MultiModalEncoderDecoderModel(nn.Module):
    def __init__(self, encoder: ImageBindModel, decoder: MMWaveDecoder, vision_dim=768, gps_dim=768, mmwave_dim=768,input_length=8, output_length=3):
        super(MultiModalEncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.input_length = input_length
        self.output_length = output_length
        
        # 计算拼接后的特征维度
        combined_feat_dim = vision_dim + gps_dim + mmwave_dim  # 256 + 128 + 384 = 768
        
        # 定义线性层，将拼接特征调整到解码器的嵌入维度
        self.proj = nn.Linear(combined_feat_dim, self.decoder.embed_dim)
    
    def forward(self, inputs, tgt_seq):
        """
        Args:
            inputs: 字典，包含 'video', 'gps', 'mmwave' 等模态的数据
            tgt_seq: [B, tgt_seq_length] 目标序列的索引
        Returns:
            output: [B, tgt_seq_length, D] 预测的 mmWave 序列
        """
        encoder_output = self.encoder(inputs)  # {ModalityType.VISION: [B, D1], ModalityType.GPS: [B, D2], ...}
        # 选择相关的模态，例如 VISION, GPS, MMWAVE
        # vision_feat = encoder_output.get(ModalityType.VISION)  # [B, D1]
        # gps_feat = encoder_output.get(ModalityType.GPS)        # [B, D2]
        # mmwave_feat = encoder_output.get(ModalityType.MMWAVE)  # [B, D3]
        vision_feat = encoder_output.get(ModalityType.VISION)  # [B, T, D1]
        gps_feat = encoder_output.get(ModalityType.GPS)        # [B, T,D2]
        mmwave_feat = encoder_output.get(ModalityType.MMWAVE)  # [B, T,D3]
        vision_feat = vision_feat[:, :self.input_length - self.output_length, :]
        gps_feat = gps_feat[:, :self.input_length - self.output_length, :]
        mmwave_feat = mmwave_feat[:, :self.input_length - self.output_length, :]
        
        # 确保所有模态的特征存在
        if vision_feat is None or gps_feat is None or mmwave_feat is None:
            raise ValueError("One or more required modalities are missing in the encoder output.")
        
        # 拼接特征
        # combined_feat = torch.cat([vision_feat, gps_feat, mmwave_feat], dim=1)  # [B, D1 + D2 + D3]
        combined_feat = torch.cat([vision_feat, gps_feat, mmwave_feat], dim=-1)  # [B, T, D1 + D2 + D3]
        
        # 使用线性层调整维度
        combined_feat = self.proj(combined_feat)  # [B, embed_dim]
        
        # 解码器预测
        output = self.decoder(combined_feat, tgt_seq)  # [B, tgt_seq_length, D]
        
        return output

class MMWaveEncoderDecoderModel(nn.Module):
    def __init__(self,encoder: ImageBindModel, decoder: MMWaveDecoder, mmwave_dim=768,input_length=8, output_length=3):
        super(MMWaveEncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.input_length = input_length
        self.output_length = output_length
        
        # 定义线性层，将 mmWave 特征调整到解码器的嵌入维度
        self.proj = nn.Linear(mmwave_dim, self.decoder.embed_dim)
    
    def forward(self, inputs, tgt_seq):
        """
        Args:
            inputs: 张量，仅包含 mmWave 特征 [B, D]
            tgt_seq: [B, tgt_seq_length] 目标序列的索引
        Returns:
            output: [B, tgt_seq_length, D] 预测的 mmWave 序列
        """
        encoder_output = self.encoder(inputs)
        
        # 选择MMWAVE模态
        mmwave_feat = encoder_output.get(ModalityType.MMWAVE)  # [B, D3]
        mmwave_feat = mmwave_feat[:, :self.input_length - self.output_length, :]
        
        # 使用线性层调整维度
        projected_feat = self.proj(mmwave_feat)  # [B, embed_dim]
        
        # 解码器预测
        output = self.decoder(projected_feat, tgt_seq)  # [B, tgt_seq_length, D]
        
        return output

class BeamVisionEncoderDecoderModel(nn.Module):
    def __init__(self, encoder: ImageBindModel, decoder: MMWaveDecoder, vision_dim=768, beam_dim=768,input_length=8, output_length=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.input_length = input_length
        self.output_length = output_length

        # 计算拼接后的特征维度
        combined_feat_dim = vision_dim + beam_dim  # 768 + 768 = 1536
        
        # 定义线性层，将拼接特征调整到解码器的嵌入维度
        self.proj = nn.Linear(combined_feat_dim, self.decoder.embed_dim)
    
    def forward(self, inputs, tgt_seq):
        """
        Args:
            inputs: 字典，包含 'video', 'beam' 等模态的数据
            tgt_seq: [B, tgt_seq_length] 目标序列的索引
        Returns:
            output: [B, tgt_seq_length, D] 预测的 mmWave 序列
        """
        encoder_output = self.encoder(inputs)
        # 选择相关的模态，例如 VISION, BEAM
        vision_feat = encoder_output.get(ModalityType.VISION)  # [B, D1]
        beam_feat = encoder_output.get(ModalityType.MMWAVE)  # [B, D2]
        vision_feat = vision_feat[:, :self.input_length - self.output_length, :]
        beam_feat = beam_feat[:, :self.input_length - self.output_length, :]

        # 确保所有模态的特征存在
        if vision_feat is None or beam_feat is None:
            raise ValueError("One or more required modalities are missing in the encoder output.")
        
        # 拼接特征
        combined_feat = torch.cat([vision_feat, beam_feat], dim=-1)  # [B, T, D1 + D2]

        # 使用线性层调整维度
        combined_feat = self.proj(combined_feat)  # [B, embed_dim]

        # 解码器预测
        output = self.decoder(combined_feat, tgt_seq)

        return output


class RawMMWaveDecoderOnlyModel(nn.Module):
    def __init__(self, decoder: MMWaveDecoder, raw_mmwave_dim=64,input_length=8, output_length=3):
        super(RawMMWaveDecoderOnlyModel, self).__init__()
        self.decoder = decoder
        
        self.input_length = input_length
        self.output_length = output_length

        # 直接将原始 mmWave 数据投影到解码器的嵌入维度
        self.proj = nn.Linear(raw_mmwave_dim, self.decoder.embed_dim)
    
    def forward(self, raw_mmwave, tgt_seq):
        """
        Args:
            raw_mmwave: 原始 mmWave 数据 [B, input_length, D_raw]
            tgt_seq: [B, tgt_seq_length] 目标序列的索引
        Returns:
            output: [B, tgt_seq_length, D] 预测的 mmWave 序列
        """
        # 假设 raw_mmwave 是时间序列数据，需要先进行池化或其他处理
        # 这里采用简单的平均池化
        # pooled_mmwave = raw_mmwave.mean(dim=1)  # [B, D_raw]
        
        # 使用线性层调整维度
        # projected_feat = self.proj(pooled_mmwave)  # [B, embed_dim]
        projected_feat = self.proj(raw_mmwave) # [B, T, D_raw] -> [B, T, embed_dim]
        
        # 解码器预测
        output = self.decoder(projected_feat, tgt_seq)  # [B, tgt_seq_length, D]
        
        return output