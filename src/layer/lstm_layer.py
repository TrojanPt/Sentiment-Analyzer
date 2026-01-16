'''
LSTM层
'''
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, device, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.dropout_rate = dropout
        
        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # 输出门参数
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        # 细胞状态候选值参数
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        
        # 层归一化
        self.layer_norm_cell = nn.LayerNorm(hidden_size)
        self.layer_norm_hidden = nn.LayerNorm(hidden_size)
        
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)

        # 初始隐藏状态及细胞状态
        self.init_h_t = nn.Parameter(torch.Tensor(self.hidden_size))
        self.init_c_t = nn.Parameter(torch.Tensor(self.hidden_size))
        
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if 'W_i' in name:
                # Xavier初始化适合线性变换
                nn.init.xavier_uniform_(param.data)
            elif 'W_h' in name:
                # 正交初始化有助于RNN稳定性
                nn.init.orthogonal_(param.data)
            elif 'b_f' in name:
                # 遗忘门偏置初始化为1，有助于长序列训练
                nn.init.ones_(param.data)
            elif 'b_' in name:
                nn.init.zeros_(param.data)
            if 'init_h_t' in name or 'init_c_t' in name:
                nn.init.normal_(param.data, mean=0.0, std=0.01)

    def _init_states(self, batch_size):
        """初始化隐藏状态和细胞状态并返回它们"""
        h_t = self.init_h_t.unsqueeze(0).expand(batch_size, -1)
        c_t = self.init_c_t.unsqueeze(0).expand(batch_size, -1)
        return h_t, c_t

    def step_forward(self, x_t, h_t, c_t, init = False) -> tuple:
        """单步前向传播"""
        if init:
            # 初始化隐藏状态和细胞状态
            h_t, c_t = self._init_states(x_t.size(0))

        # 应用dropout到输入
        x_t = self.dropout(x_t)
        
        # 输入门计算
        i_t = torch.sigmoid(
            F.linear(x_t, self.W_ii.T, self.b_i) + 
            F.linear(h_t, self.W_hi.T)
        )
        
        # 遗忘门计算
        f_t = torch.sigmoid(
            F.linear(x_t, self.W_if.T, self.b_f) + 
            F.linear(h_t, self.W_hf.T) 
        )
        
        # 输出门计算
        o_t = torch.sigmoid(
            F.linear(x_t, self.W_io.T, self.b_o) + 
            F.linear(h_t, self.W_ho.T)
        )
        
        # 候选细胞状态计算
        g_t = torch.tanh(
            F.linear(x_t, self.W_ig.T, self.b_g) + 
            F.linear(h_t, self.W_hg.T)
        )
        
        # 更新细胞状态（添加层归一化）
        new_c_t = f_t * c_t + i_t * g_t
        new_c_t = self.layer_norm_cell(new_c_t)
        
        # 更新隐藏状态（添加层归一化）
        new_h_t = o_t * torch.tanh(new_c_t)
        new_h_t = self.layer_norm_hidden(new_h_t)
        
        return new_h_t, new_c_t

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播"""
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态和单元状态
        h_t, c_t = self._init_states(batch_size)
        
        lstm_outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.step_forward(x_t, h_t, c_t)
            lstm_outputs.append(h_t)

        # 返回序列中每个时间步的输出和最终状态
        lstm_outputs = torch.stack(lstm_outputs)
        
        return lstm_outputs, (h_t, c_t)


class BiLSTMWrapper(nn.Module):
    """双向LSTM封装器，使用两个LSTMLayer"""
    def __init__(self, input_size, hidden_size, device, dropout=0.2):
        super().__init__()
        self.forward_lstm = LSTMLayer(input_size, hidden_size, device, dropout)
        self.backward_lstm = LSTMLayer(input_size, hidden_size, device, dropout)
        self.device = device
        
    def forward(self, x):
        """
        前向传播，同时处理正向和反向序列
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            
        Returns:
            outputs: 双向LSTM输出 [batch_size, seq_len, hidden_size*2]
            (h_n, c_n): 最终隐藏状态和细胞状态 [(batch_size, hidden_size*2), (batch_size, hidden_size*2)]
        """
        # 正向LSTM
        forward_outputs, (h_f, c_f) = self.forward_lstm(x)
        
        # 反向LSTM（翻转输入序列）
        reversed_x = x.flip(dims=[1])
        backward_outputs, (h_b, c_b) = self.backward_lstm(reversed_x)
        
        # 翻转回来以匹配正向序列
        backward_outputs = backward_outputs.flip(dims=[0])
        
        # 合并正向和反向输出 [seq_len, batch_size, hidden_size*2]
        outputs = torch.cat((forward_outputs, backward_outputs), dim=2)
        
        # 合并最终隐藏状态和细胞状态
        h_n = torch.cat((h_f, h_b), dim=1)
        c_n = torch.cat((c_f, c_b), dim=1)
        
        return outputs, (h_n, c_n)


class StackedLSTM(nn.Module):
    """堆叠多层LSTM"""
    def __init__(
            self, 
            input_size: int,
            hidden_size: int,
            num_layers: int,
            device: torch.device,
            dropout: float = 0.2,
            bidirectional: bool = False
            ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # 创建LSTM层堆栈
        self.lstm_layers = nn.ModuleList()
        
        # 第一层使用原始输入大小
        if bidirectional:
            self.lstm_layers.append(BiLSTMWrapper(input_size, hidden_size, device, dropout))
            # 后续层的输入是前一层的双向输出，因此是hidden_size*2
            for _ in range(1, num_layers):
                self.lstm_layers.append(BiLSTMWrapper(hidden_size*2, hidden_size, device, dropout))
        else:
            self.lstm_layers.append(LSTMLayer(input_size, hidden_size, device, dropout))
            # 后续层使用前一层的输出尺寸
            for _ in range(1, num_layers):
                self.lstm_layers.append(LSTMLayer(hidden_size, hidden_size, device, dropout))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            
        Returns:
            outputs: 最后一层LSTM的输出序列
            hidden_states: 所有层的最终隐藏状态和细胞状态
        """
        batch_size, seq_len, _ = x.size()
        hidden_states = []
        current_input = x
        
        # 逐层处理
        for i, lstm_layer in enumerate(self.lstm_layers):
            outputs, (h_n, c_n) = lstm_layer(current_input)
            
            # 重塑输出以用作下一层的输入
            # outputs的形状是[seq_len, batch_size, hidden_size]或[seq_len, batch_size, hidden_size*2]
            current_input = outputs.transpose(0, 1)  # 变为[batch_size, seq_len, hidden_size(*2)]
            
            # 存储此层的最终状态
            hidden_states.append((h_n, c_n))
        
        return outputs, hidden_states

