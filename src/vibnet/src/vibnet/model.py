"""
神经网络模型定义
用于预测单自由度系统的振动响应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VibrationNet(nn.Module):
    """
    前馈神经网络，用于预测单自由度系统的振动响应
    
    输入: 系统参数 (质量、阻尼比) + 激励力序列统计特征
    输出: 位移响应时间序列
    """
    
    def __init__(self, input_size: int, hidden_sizes: list[int], 
                 output_size: int, dropout_rate: float = 0.1):
        """
        初始化神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层神经元数量列表
            output_size: 输出序列长度
            dropout_rate: Dropout比率
        """
        super(VibrationNet, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        # 隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_size]
            
        Returns:
            预测的位移响应序列 [batch_size, output_size]
        """
        return self.network(x)


class ImprovedVibrationNet(nn.Module):
    """
    改进的振动响应预测网络
    包含残差连接和注意力机制
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 output_size: int = 100, num_layers: int = 3):
        super(ImprovedVibrationNet, self).__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_layers)
        ])
        
        # 注意力层
        self.attention = SelfAttention(hidden_size)
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入投影
        x = F.relu(self.input_proj(x))
        
        # 残差块
        for block in self.res_blocks:
            x = block(x)
        
        # 注意力
        x = self.attention(x.unsqueeze(1)).squeeze(1)
        
        # 输出投影
        return self.output_proj(x)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, hidden_size: int):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = F.relu(self.norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.norm2(self.layer2(x))
        
        return F.relu(x + residual)


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, hidden_size: int):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, hidden_size]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 计算注意力分数
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        attended = torch.bmm(attn_weights, v)
        
        return attended
