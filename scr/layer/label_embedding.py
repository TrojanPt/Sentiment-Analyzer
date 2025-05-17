from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelEmbedding(nn.Module):
    """标签嵌入"""
    
    def __init__(self, labels: List[str], embedding_model):
        """
        初始化标签嵌入
        
        Args:
            labels (List[str]): 标签列表
            embedding_model: 嵌入模型，用于获取初始标签向量
        """
        super().__init__()
        self.labels = labels
        self.num_labels = len(labels)
        self.embedding_dim = embedding_model.embedding_dim
        
        # 使用嵌入模型获取标签的初始向量表示
        initial_embeddings = []
        for label in labels:
            # 获取标签文本的嵌入向量
            label_embed = embedding_model.embed(label)
            initial_embeddings.append(label_embed)
        
        # 将初始嵌入堆叠为张量
        initial_embeddings = torch.stack(initial_embeddings)
        
        # 创建嵌入参数
        self.label_embeddings = nn.Parameter(initial_embeddings)
        
    def forward(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        计算模型输出与每个标签嵌入的相似度
        
        Args:
            model_output: 模型输出，形状为 [batch_size, embedding_dim]
            
        Returns:
            形状为 [batch_size, num_labels] 的相似度分数
        """
        # 归一化标签嵌入和模型输出
        normalized_labels = F.normalize(self.label_embeddings, p=2, dim=1)
        normalized_output = F.normalize(model_output, p=2, dim=1)
        
        # 计算余弦相似度 (点积)
        logits = torch.matmul(normalized_output, normalized_labels.t())
        
        # 应用温度系数使分布更锐利
        temperature = 0.07
        return logits / temperature
    
    def get_label_embeddings(self) -> torch.Tensor:
        """获取当前标签嵌入"""
        return self.label_embeddings