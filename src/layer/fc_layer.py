'''
全连接层模块
'''

import torch
import torch.nn as nn


class FCLayer(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # 构建多层网络
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),  # 层归一化
                nn.SiLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.fc_layers = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对于隐藏层，使用Kaiming初始化
                if m.out_features != 1:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity="leaky_relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
    def forward(self, x):
        output = self.fc_layers(x)
        
        # L2正则化损失
        l2_reg_loss = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg_loss += torch.sum(param ** 2)
        
        return output, l2_reg_loss

