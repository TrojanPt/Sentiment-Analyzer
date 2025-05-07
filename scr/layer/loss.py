
# 使用标签平滑正则化的交叉熵损失
import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(-1, target.argmax(dim=-1, keepdim=True), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
