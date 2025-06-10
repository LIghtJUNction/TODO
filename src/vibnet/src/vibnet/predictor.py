"""
模型预测和结果可视化
"""

import torch
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def predict_response(model: torch.nn.Module, mass: float, damping_ratio: float,
                    force: np.ndarray, device: torch.device,
                    force_features: Optional[np.ndarray] = None) -> np.ndarray:
    """
    使用训练好的模型预测振动响应
    
    Args:
        model: 训练好的模型
        mass: 质量
        damping_ratio: 阻尼比
        force: 激励力时间序列
        device: 计算设备
        force_features: 预计算的力特征（可选）
        
    Returns:
        预测的位移响应
    """
    model.eval()
    
    # 提取激励力特征
    if force_features is None:
        from data_generator import extract_force_features
        force_features = extract_force_features(force)
    
    # 假设固有频率（在实际应用中应该是已知参数或者作为输入）
    natural_freq = 10.0  # 默认值，可以根据需要调整
    
    # 组合输入特征
    system_params = np.array([mass, damping_ratio, natural_freq], dtype=np.float32)
    input_features = np.concatenate([system_params, force_features])
    
    # 加载标准化参数
    try:
        norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
        inputs_mean = norm_params['inputs_mean']
        inputs_std = norm_params['inputs_std']
        outputs_mean = norm_params['outputs_mean']
        outputs_std = norm_params['outputs_std']
        
        # 标准化输入
        input_features = (input_features - inputs_mean) / inputs_std
        
    except FileNotFoundError:
        print("警告: 未找到标准化参数文件，使用原始输入")
        outputs_mean, outputs_std = 0.0, 1.0
    
    # 转换为张量并预测
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_features).float().unsqueeze(0).to(device)
        prediction = model(input_tensor)
        prediction = prediction.cpu().numpy().squeeze()
    
    # 反标准化输出
    prediction = prediction * outputs_std + outputs_mean
    
    return prediction


def plot_results(time: np.ndarray, true_response: np.ndarray, 
                predicted_response: np.ndarray, force: np.ndarray,
                save_path: Optional[str] = None):
    """
    绘制预测结果对比图
    
    Args:
        time: 时间向量
        true_response: 真实响应
        predicted_response: 预测响应
        force: 激励力
        save_path: 保存路径（可选）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('单自由度系统振动响应预测结果', fontsize=16, fontweight='bold')
    
    # 激励力
    axes[0].plot(time, force, 'g-', linewidth=1.5, label='激励力')
    axes[0].set_ylabel('力 (N)', fontsize=12)
    axes[0].set_title('激励力时间历程', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 位移响应对比
    axes[1].plot(time, true_response, 'b-', linewidth=2, label='真实响应', alpha=0.8)
    axes[1].plot(time, predicted_response, 'r--', linewidth=2, label='预测响应', alpha=0.8)
    axes[1].set_ylabel('位移 (m)', fontsize=12)
    axes[1].set_title('位移响应对比', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 误差分析
    error = predicted_response - true_response
    axes[2].plot(time, error, 'k-', linewidth=1.5, label='预测误差')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('时间 (s)', fontsize=12)
    axes[2].set_ylabel('误差 (m)', fontsize=12)
    axes[2].set_title('预测误差', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # 计算误差统计
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # 添加误差统计文本
    error_text = f'MAE: {mae:.6f} m\nRMSE: {rmse:.6f} m\nMax Error: {max_error:.6f} m'
    axes[2].text(0.02, 0.98, error_text, transform=axes[2].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果图已保存到: {save_path}")
    
    plt.show()
    
    # 打印数值统计
    print("\n=== 预测精度统计 ===")
    print(f"平均绝对误差 (MAE): {mae:.6f} m")
    print(f"均方根误差 (RMSE): {rmse:.6f} m")
    print(f"最大误差: {max_error:.6f} m")
    print(f"相对误差 (MAE/真实值标准差): {mae/np.std(true_response)*100:.2f}%")


def evaluate_model(model: torch.nn.Module, test_data_loader, device: torch.device):
    """
    评估模型在测试集上的性能
    
    Args:
        model: 训练好的模型
        test_data_loader: 测试数据加载器
        device: 计算设备
    """
    model.eval()
    
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            predictions = model(inputs)
            
            # 计算误差
            mae = torch.mean(torch.abs(predictions - targets))
            rmse = torch.sqrt(torch.mean((predictions - targets)**2))
            
            total_mae += mae.item() * inputs.size(0)
            total_rmse += rmse.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # 收集所有预测和目标值
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 计算总体统计
    avg_mae = total_mae / total_samples
    avg_rmse = total_rmse / total_samples
    
    print("\n=== 模型测试集性能 ===")
    print(f"样本数量: {total_samples}")
    print(f"平均绝对误差 (MAE): {avg_mae:.6f}")
    print(f"均方根误差 (RMSE): {avg_rmse:.6f}")
    
    # 将结果转换为numpy数组
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_mae, avg_rmse, all_predictions, all_targets


def compare_different_systems():
    """
    比较不同系统参数下的预测效果
    """
    from model import VibrationNet
    from data_generator import generate_random_force, sdof_response
    
    # 加载训练好的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VibrationNet(input_size=12, hidden_sizes=[64, 128, 64], output_size=100)
    
    try:
        model.load_state_dict(torch.load('vibration_model.pth', map_location=device))
        model.to(device)
        print("模型加载成功")
    except FileNotFoundError:
        print("未找到训练好的模型文件")
        return
    
    # 定义不同的系统参数
    test_cases = [
        {"mass": 1.0, "damping": 0.02, "desc": "轻阻尼系统"},
        {"mass": 1.0, "damping": 0.1, "desc": "中等阻尼系统"},
        {"mass": 2.0, "damping": 0.05, "desc": "重质量系统"},
        {"mass": 0.5, "damping": 0.05, "desc": "轻质量系统"},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, case in enumerate(test_cases):
        # 生成测试激励
        force = generate_random_force(100, 0.01, "sine")
        
        # 真实响应
        true_response = sdof_response(
            force, case["mass"], case["damping"], natural_freq=10.0, dt=0.01
        )
        
        # 预测响应
        predicted_response = predict_response(
            model, case["mass"], case["damping"], force, device
        )
        
        # 绘制对比
        time = np.linspace(0, 1, 100)
        axes[i].plot(time, true_response[:100], 'b-', label='真实', linewidth=2)
        axes[i].plot(time, predicted_response[:100], 'r--', label='预测', linewidth=2)
        axes[i].set_title(f'{case["desc"]} (m={case["mass"]}, ζ={case["damping"]})')
        axes[i].set_xlabel('时间 (s)')
        axes[i].set_ylabel('位移 (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # 计算误差
        mae = np.mean(np.abs(predicted_response[:100] - true_response[:100]))
        axes[i].text(0.02, 0.98, f'MAE: {mae:.4f}', transform=axes[i].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig('system_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 运行系统对比分析
    compare_different_systems()
