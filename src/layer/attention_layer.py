
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """自注意力机制层"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 定义注意力计算所需的权重矩阵
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 初始化
        nn.init.xavier_uniform_(self.attention_weights.data)
        
    def forward(self, lstm_output):
        """
        计算注意力权重并应用于LSTM输出
        
        Args:
            lstm_output: LSTM层的输出 [batch_size, seq_len, hidden_size]
            
        Returns:
            context: 注意力加权后的上下文向量 [batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        batch_size, seq_len, _ = lstm_output.size()
        
        # 应用层归一化
        normalized_output = self.layer_norm(lstm_output)
        
        # 计算注意力分数: [batch_size, seq_len, 1]
        attention_scores = torch.matmul(normalized_output, self.attention_weights)
        
        # 应用softmax获取注意力权重: [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 使用注意力权重计算上下文向量: [batch_size, hidden_size]
        context = torch.sum(lstm_output * attention_weights, dim=1)
        
        # 返回上下文向量和注意力权重
        return context, attention_weights.squeeze(2)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_size = hidden_size // num_heads
        
        # 定义多头线性投影
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # 定义输出层和正则化
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        

        # 全局注意力参数（AttentionLayer）
        # 定义注意力计算所需的权重矩阵
        self.global_attention_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.layer_norm = nn.LayerNorm(hidden_size)
        # 初始化
        nn.init.xavier_uniform_(self.global_attention_weights.data)


    def forward(self, x):
        batch_size = x.size(0)
        
        # 线性投影并分头 [B, L, H] -> [B, L, num_heads, head_size]
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_size)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_size)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_size)
        
        # 转置为 [B, num_heads, L, head_size]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_size)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力到值向量
        context = torch.matmul(attention_weights, v)    # [B, nh, L, hs]
        
        # 拼接多头结果并线性变换 [B, L, H]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.output_linear(context)

        # 全局注意力池化
        # 层归一化
        normalized_output = self.layer_norm(output)     # [B, L, H]
        
        # 计算全局注意力分数
        attention_scores = torch.matmul(normalized_output, self.global_attention_weights)   # [B, L, 1]
        global_attention = F.softmax(attention_scores, dim=1)   # [B, L, 1]
        
        # 生成上下文向量
        final_context = torch.sum(output * global_attention, dim=1)     # [B, H]
        
        return final_context, global_attention.squeeze(2)   # [B, H], [B, L]