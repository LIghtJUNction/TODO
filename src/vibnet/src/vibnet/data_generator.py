"""
单自由度系统振动数据生成器
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple
from scipy.integrate import odeint


def sdof_response(force: np.ndarray, mass: float = 1.0, 
                  damping_ratio: float = 0.05, natural_freq: float = 10.0,
                  dt: float = 0.01) -> np.ndarray:
    """
    计算单自由度系统在给定激励下的位移响应
    
    系统方程: m*x'' + c*x' + k*x = F(t)
    其中: c = 2*zeta*sqrt(k*m), k = m*omega_n^2
    
    Args:
        force: 激励力时间序列
        mass: 质量
        damping_ratio: 阻尼比
        natural_freq: 固有频率 (rad/s)
        dt: 时间步长
        
    Returns:
        位移响应时间序列
    """
    # 计算系统参数
    k = mass * (natural_freq ** 2)  # 刚度
    c = 2 * damping_ratio * np.sqrt(k * mass)  # 阻尼系数
    
    def system_ode(y, t, force_interp):
        """系统微分方程"""
        x, x_dot = y
        # 插值获取当前时刻的激励力
        f_t = force_interp(t) if callable(force_interp) else 0
        
        x_ddot = (f_t - c * x_dot - k * x) / mass
        return [x_dot, x_ddot]
    
    # 时间向量
    t = np.arange(0, len(force) * dt, dt)
    
    # 创建力的插值函数
    from scipy.interpolate import interp1d
    if len(force) > 1:
        force_interp = interp1d(t[:len(force)], force, 
                               kind='linear', bounds_error=False, fill_value=0)
    else:
        force_interp = lambda t: force[0] if len(force) > 0 else 0
    
    # 初始条件 [位移, 速度]
    y0 = [0.0, 0.0]
    
    # 求解微分方程
    solution = odeint(system_ode, y0, t, args=(force_interp,))
    
    return solution[:, 0]  # 返回位移


def generate_random_force(length: int, dt: float = 0.01, 
                         force_type: str = "mixed") -> np.ndarray:
    """
    生成随机激励力
    
    Args:
        length: 序列长度
        dt: 时间步长
        force_type: 激励类型 ("sine", "impulse", "random", "chirp", "mixed")
        
    Returns:
        激励力时间序列
    """
    t = np.arange(0, length * dt, dt)[:length]
    
    if force_type == "sine":
        # 正弦激励
        freq = np.random.uniform(0.5, 5.0)  # 随机频率
        amplitude = np.random.uniform(50, 200)
        force = amplitude * np.sin(2 * np.pi * freq * t)
        
    elif force_type == "impulse":
        # 脉冲激励
        force = np.zeros(length)
        impulse_pos = np.random.randint(5, length // 4)
        impulse_width = np.random.randint(1, 5)
        impulse_magnitude = np.random.uniform(100, 500)
        force[impulse_pos:impulse_pos + impulse_width] = impulse_magnitude
        
    elif force_type == "random":
        # 随机激励 (白噪声 + 低频成分)
        force = np.random.normal(0, 30, length)
        # 添加低频成分
        low_freq = np.random.uniform(0.1, 1.0)
        force += 50 * np.sin(2 * np.pi * low_freq * t)
        
    elif force_type == "chirp":
        # 扫频激励
        f0 = np.random.uniform(0.1, 1.0)
        f1 = np.random.uniform(2.0, 8.0)
        amplitude = np.random.uniform(80, 150)
        from scipy.signal import chirp
        force = amplitude * chirp(t, f0, t[-1], f1)
        
    elif force_type == "mixed":
        # 混合激励
        force_types = ["sine", "impulse", "random", "chirp"]
        selected_type = np.random.choice(force_types)
        force = generate_random_force(length, dt, selected_type)
        
    else:
        raise ValueError(f"未知的激励类型: {force_type}")
    
    return force


def extract_force_features(force: np.ndarray) -> np.ndarray:
    """
    从激励力时间序列中提取特征
    
    Args:
        force: 激励力时间序列
        
    Returns:
        特征向量
    """
    features = []
    
    # 统计特征
    features.extend([
        np.mean(force),           # 均值
        np.std(force),            # 标准差
        np.max(force),            # 最大值
        np.min(force),            # 最小值
        np.ptp(force),            # 峰峰值
    ])
    
    # 能量特征
    features.extend([
        np.sum(force ** 2),       # 总能量
        np.mean(force ** 2),      # 平均功率
    ])
    
    # 频域特征 (简化)
    fft = np.fft.rfft(force)
    power_spectrum = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(force))
    
    # 主频率
    dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # 排除直流分量
    features.append(freqs[dominant_freq_idx] if len(freqs) > dominant_freq_idx else 0)
    
    # 频谱质心
    if np.sum(power_spectrum) > 0:
        spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
    else:
        spectral_centroid = 0
    features.append(spectral_centroid)
    
    return np.array(features, dtype=np.float32)


class VibrationDataset(Dataset):
    """振动响应数据集"""
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Args:
            inputs: 输入特征 [n_samples, n_features]
            outputs: 输出响应 [n_samples, sequence_length]
        """
        self.inputs = torch.from_numpy(inputs).float()
        self.outputs = torch.from_numpy(outputs).float()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def generate_sdof_data(n_samples: int = 1000, 
                       val_split: float = 0.2,
                       sequence_length: int = 100,
                       dt: float = 0.01) -> Tuple[DataLoader, DataLoader]:
    """
    生成单自由度系统振动数据
    
    Args:
        n_samples: 样本数量
        val_split: 验证集比例
        sequence_length: 时间序列长度
        dt: 时间步长
        
    Returns:
        训练和验证数据加载器
    """
    print(f"生成 {n_samples} 个样本...")
    
    inputs = []
    outputs = []
    for i in range(n_samples):
        if (i + 1) % 20 == 0:  # 更频繁的进度更新
            print(f"已生成 {i + 1}/{n_samples} 个样本")
        
        # 随机系统参数
        mass = np.random.uniform(0.5, 2.0)
        damping_ratio = np.random.uniform(0.01, 0.2)
        natural_freq = np.random.uniform(5.0, 20.0)
        
        # 生成激励力
        force = generate_random_force(sequence_length, dt)
        
        # 计算响应
        response = sdof_response(force, mass, damping_ratio, natural_freq, dt)
        
        # 提取特征
        force_features = extract_force_features(force)
        system_params = np.array([mass, damping_ratio, natural_freq], dtype=np.float32)
        
        # 组合输入特征
        input_features = np.concatenate([system_params, force_features])
        
        inputs.append(input_features)
        outputs.append(response[:sequence_length])  # 确保长度一致
    
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    print(f"输入特征维度: {inputs.shape}")
    print(f"输出响应维度: {outputs.shape}")
    
    # 数据标准化
    inputs_mean = np.mean(inputs, axis=0)
    inputs_std = np.std(inputs, axis=0)
    inputs_std[inputs_std == 0] = 1  # 避免除零
    inputs = (inputs - inputs_mean) / inputs_std
    
    outputs_mean = np.mean(outputs)
    outputs_std = np.std(outputs)
    outputs = (outputs - outputs_mean) / outputs_std
    
    # 保存标准化参数（实际应用中需要保存用于预测时的反标准化）
    norm_params = {
        'inputs_mean': inputs_mean,
        'inputs_std': inputs_std,
        'outputs_mean': outputs_mean,
        'outputs_std': outputs_std
    }
    np.save('normalization_params.npy', norm_params)
    
    # 创建数据集
    dataset = VibrationDataset(inputs, outputs)
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    return train_loader, val_loader
