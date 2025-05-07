
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
