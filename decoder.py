# decoder.py

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class MMWaveDecoder(nn.Module):
    def __init__(self, embed_dim=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, output_dim=64, max_seq_length=50):
        super(MMWaveDecoder, self).__init__()
        decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 目标序列的嵌入（学习的嵌入）
        self.target_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # 输出预测 mmWave 数据
        self.output_layer = nn.Linear(embed_dim, output_dim)
        
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length

    def forward(self, memory, tgt_seq):
        """
        Args:
            memory: [B, T, embed_dim] 编码器输出的特征
            tgt_seq: [B, tgt_seq_length] 目标序列的索引
        Returns:
            output: [B, tgt_seq_length, output_dim] 预测的 mmWave 序列
        """
        B, tgt_seq_length = tgt_seq.size()
        # 目标嵌入
        tgt_embeddings = self.target_embedding(tgt_seq) * (self.embed_dim ** 0.5)  # [B, tgt_seq_length, embed_dim]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)  # [tgt_seq_length, B, embed_dim]
        
        # 构建 memory
        # memory 仍保持 [B, T, embed_dim]，不做 unsqueeze，符合 Transformer 解码器要求的 [T, B, embed_dim] 输入
        memory = memory.transpose(0, 1)  # [T, B, embed_dim]
        
        # 创建 tgt_mask
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_length).to(memory.device)  # [tgt_seq_length, tgt_seq_length]
        
        # 前向传播 Transformer 解码器
        decoder_output = self.transformer_decoder(tgt_embeddings, memory, tgt_mask=tgt_mask)  # [tgt_seq_length, B, embed_dim]
        
        decoder_output = decoder_output.transpose(0, 1)  # [B, tgt_seq_length, embed_dim]
        output = self.output_layer(decoder_output)  # [B, tgt_seq_length, output_dim]
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
