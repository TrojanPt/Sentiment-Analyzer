
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用标签平滑正则化的交叉熵损失
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, reduction='mean', gamma=0.0):
        """
        标签平滑损失，可处理类别不平衡问题
        
        Args:
            classes: 类别数
            smoothing: 平滑参数
            reduction: 损失计算方式 ('mean', 'sum')
            gamma: Focal Loss的gamma参数，默认为0 (不使用Focal Loss)
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.reduction = reduction
        self.gamma = gamma
        
        if reduction not in ['mean', 'sum']:
            raise ValueError("Reduction must be 'mean' or 'sum'")
            
    def forward(
            self, 
            pred: torch.Tensor, 
            target: torch.Tensor,
            class_weights: Optional[torch.Tensor] = None
        ):
        '''
        Args:
            pred: 预测值，形状为 (batch_size, num_classes)
            target: 目标值，形状为 (batch_size, num_classes)
            class_weights: 类别权重，形状为 (num_classes,)
        '''
        probs = F.softmax(pred, dim=-1)
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 获取目标类别索引
        target_idx = target.argmax(dim=-1, keepdim=True)

        # 如果没有提供类别权重，则使用均等权重
        if class_weights is None:
            class_weights = torch.ones(self.classes, dtype=pred.dtype, device=pred.device)
        
        with torch.no_grad():
            # 创建平滑标签分布
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(-1, target_idx, self.confidence)
            
            # 获取每个样本对应的类别权重
            sample_weights = torch.gather(class_weights.expand(pred.size(0), -1), 
                                          1, target_idx).squeeze()
        
        # 计算交叉熵损失
        loss = -true_dist * log_probs
        
        # 应用Focal Loss来减少易分类样本的权重
        if self.gamma > 0:
            # 获取目标类别的预测概率
            pt = torch.gather(probs, 1, target_idx)
            focal_weight = (1 - pt).pow(self.gamma)
            loss = loss * focal_weight
        
        # 对每个样本求和并应用类别权重
        loss = torch.sum(loss, dim=-1) * sample_weights
        
        # 应用reduction
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.sum(loss) / sample_weights.sum()

# 使用语义加权的交叉熵损失
class SemanticWeightedLoss(nn.Module):
    """
    语义加权损失：根据标签间的语义相似性调整损失权重
    
    语义相似的标签错误预测会得到较小的惩罚
    语义差异大的标签错误预测会得到较大的惩罚
    """
    def __init__(
            self, 
            classes: int, 
            smoothing: float = 0.1, 
            reduction: str = 'mean', 
            gamma: float = 0.0,
            semantic_weight: float = 0.7
        ):
        '''
        Args:
            classes: 类别数
            smoothing: 平滑参数
            reduction: 损失计算方式 ('mean', 'sum')
            gamma: Focal Loss的gamma参数，默认为0 (不使用Focal Loss)
            semantic_weight: 语义相似度权重因子
        '''
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.reduction = reduction
        self.gamma = gamma
        self.semantic_weight = semantic_weight
        
        if reduction not in ['mean', 'sum']:
            raise ValueError("Reduction must be 'mean' or 'sum'")
    
    def forward(
            self, 
            pred: torch.Tensor,  # 模型输出的logits
            target: torch.Tensor,  # one-hot目标标签
            label_embeddings: torch.Tensor,  # 标签嵌入向量 [num_classes, embedding_dim]
            class_weights: Optional[torch.Tensor] = None  # 类别权重
        ):
        """
        计算语义加权损失
        
        Args:
            pred: 预测值，形状为 (batch_size, num_classes)
            target: 目标值，形状为 (batch_size, num_classes)
            label_embeddings: 标签嵌入向量，形状为 (num_classes, embedding_dim)
            class_weights: 类别权重，形状为 (num_classes,)
        """
        batch_size = pred.size(0)
        probs = F.softmax(pred, dim=-1)
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 获取目标类别索引
        target_idx = target.argmax(dim=-1, keepdim=True)
        true_idx = target_idx.squeeze(1) 
        
        # 如果没有提供类别权重，则使用均等权重
        if class_weights is None:
            class_weights = torch.ones(self.classes, dtype=pred.dtype, device=pred.device)
        
        # 计算标签间的语义相似度矩阵
        normalized_embeddings = F.normalize(label_embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.t())  # [num_classes, num_classes]
        
        with torch.no_grad():
            # 创建平滑标签分布
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(-1, target_idx, self.confidence)
            
            # 获取每个样本对应的类别权重
            sample_weights = torch.gather(class_weights.expand(batch_size, -1), 
                                          1, target_idx).squeeze()
            
            # 计算语义权重矩阵 [batch_size, num_classes]
            current_similarities = similarity_matrix[true_idx]  # 向量化操作
            semantic_weights = 1.0 - self.semantic_weight * current_similarities
            
            # 确保semantic_weights始终是二维的 [batch_size, num_classes]
            if len(semantic_weights.shape) == 1:
                semantic_weights = semantic_weights.unsqueeze(0)

            # 将正确类别的权重设为1.0
            mask = torch.zeros_like(semantic_weights, dtype=torch.bool)
            mask.scatter_(1, target_idx, True)
            semantic_weights[mask] = 1.0
        
        # 计算交叉熵损失
        base_loss = -true_dist * log_probs
        
        # 应用语义权重
        loss = base_loss * semantic_weights
        
        # 应用Focal Loss
        if self.gamma > 0:
            pt = torch.gather(probs, 1, target_idx)  # [batch_size, 1]
            focal_weight = (1 - pt).pow(self.gamma)
            loss = loss * focal_weight  # 广播乘法
        
        # 对每个样本求和并应用类别权重
        loss = torch.sum(loss, dim=-1) * sample_weights
        
        # 应用reduction
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            # return torch.sum(loss) / torch.sum(sample_weights)    # 使用sample_weights进行平均
            return torch.sum(loss) / batch_size    # 使用batch_size进行平均

if __name__ == '__main__':
    # 测试SemanticWeightedLoss
    num_classes = 3
    embedding_dim = 5
    batch_size = 2

    # 设置随机种子
    torch.manual_seed(0.721)

    # 随机生成预测值、目标值和标签嵌入
    pred = torch.Tensor([[-0.2, 0.5, 0.3],
                         [0.1, -0.3, 0.2]])
    target = torch.Tensor([[0, 1, 0],
                          [1, 0, 0]])
    label_embeddings = torch.randn(num_classes, embedding_dim)

    class_weights = torch.Tensor([1.0, 2.0, 3.0])  # 类别权重

    # 初始化损失函数
    loss_fn = SemanticWeightedLoss(
        classes=num_classes, 
        smoothing=0.1, 
        reduction='mean', 
        gamma=2.0, 
        semantic_weight=0.7
    )

    # 计算损失
    loss = loss_fn(pred, target, label_embeddings, class_weights)

    total_loss = loss * batch_size

    # 打印结果
    print("Predictions:", pred)
    print("Target:", target)
    print("Label Embeddings:", label_embeddings)
    print("Loss:", total_loss)

    pred_1 = torch.Tensor([[-0.2, 0.5, 0.3]])
    target_1 = torch.Tensor([[0, 1, 0]])
    loss_1 = loss_fn(pred_1, target_1, label_embeddings, class_weights)
    print("Loss_1:", loss_1)
    pred_2 = torch.Tensor([[0.1, -0.3, 0.2]])
    target_2 = torch.Tensor([[1, 0, 0]])
    loss_2 = loss_fn(pred_2, target_2, label_embeddings, class_weights)
    print("Loss_2:", loss_2)
    print("Loss_1 + Loss_2:", loss_1 + loss_2)
    print("Is Loss_1 + Loss_2 equal to Loss:", torch.isclose(loss_1 + loss_2, total_loss))