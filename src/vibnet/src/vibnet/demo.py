"""
简化的演示脚本：快速测试振动响应预测
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import VibrationNet
from data_generator import generate_random_force, sdof_response, extract_force_features


def quick_demo():
    """快速演示脚本"""
    print("=== 快速演示：单自由度系统振动响应预测 ===\n")
    
    # 设置设备和随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"使用设备: {device}")
    
    # 1. 创建简单的神经网络
    print("1. 创建神经网络模型...")
    model = VibrationNet(
        input_size=12,  # 系统参数(3) + 力特征(9)
        hidden_sizes=[32, 64, 32],
        output_size=50,  # 简化的序列长度
        dropout_rate=0.1
    ).to(device)
    
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 生成少量训练数据
    print("2. 生成训练数据...")
    n_samples = 200  # 减少样本数量以便快速演示
    sequence_length = 50
    dt = 0.02
    
    inputs = []
    outputs = []
    
    for i in range(n_samples):
        # 随机系统参数
        mass = np.random.uniform(0.8, 1.2)
        damping_ratio = np.random.uniform(0.02, 0.1)
        natural_freq = np.random.uniform(8.0, 12.0)
        
        # 生成激励力（简化）
        force = generate_random_force(sequence_length, dt, "sine")
        
        # 计算真实响应
        response = sdof_response(force, mass, damping_ratio, natural_freq, dt)
        
        # 提取特征
        force_features = extract_force_features(force)
        system_params = np.array([mass, damping_ratio, natural_freq])
        input_features = np.concatenate([system_params, force_features])
        
        inputs.append(input_features)
        outputs.append(response[:sequence_length])
    
    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    
    # 简单标准化
    inputs = (inputs - np.mean(inputs, axis=0)) / (np.std(inputs, axis=0) + 1e-8)
    outputs = (outputs - np.mean(outputs)) / (np.std(outputs) + 1e-8)
    
    print(f"   训练数据形状: inputs {inputs.shape}, outputs {outputs.shape}")
    
    # 3. 快速训练
    print("3. 快速训练模型...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # 转换为张量
    inputs_tensor = torch.from_numpy(inputs).to(device)
    outputs_tensor = torch.from_numpy(outputs).to(device)
    
    # 简单的训练循环
    losses = []
    for epoch in range(50):  # 只训练50个epoch
        optimizer.zero_grad()
        predictions = model(inputs_tensor)
        loss = criterion(predictions, outputs_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/50, Loss: {loss.item():.6f}")
    
    # 4. 测试预测
    print("4. 测试预测效果...")
    model.eval()
    
    # 创建测试样本
    test_mass = 1.0
    test_damping = 0.05
    test_natural_freq = 10.0
    test_force = 100 * np.sin(np.linspace(0, 2*np.pi, sequence_length))
    
    # 真实响应
    true_response = sdof_response(test_force, test_mass, test_damping, test_natural_freq, dt)
    
    # 预测响应
    test_force_features = extract_force_features(test_force)
    test_system_params = np.array([test_mass, test_damping, test_natural_freq])
    test_input = np.concatenate([test_system_params, test_force_features])
    
    # 使用相同的标准化
    test_input = (test_input - np.mean(inputs_tensor.cpu().numpy(), axis=0)) / (np.std(inputs_tensor.cpu().numpy(), axis=0) + 1e-8)
    
    with torch.no_grad():
        test_input_tensor = torch.from_numpy(test_input).float().unsqueeze(0).to(device)
        predicted_response = model(test_input_tensor).cpu().numpy().squeeze()
    
    # 反标准化
    predicted_response = predicted_response * (np.std(outputs) + 1e-8) + np.mean(outputs)
    true_response = true_response[:sequence_length]
    
    # 5. 可视化结果
    print("5. 绘制结果...")
    
    time = np.linspace(0, sequence_length * dt, sequence_length)
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: 激励力
    plt.subplot(3, 1, 1)
    plt.plot(time, test_force, 'g-', linewidth=2, label='激励力')
    plt.ylabel('力 (N)')
    plt.title('输入激励力')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图2: 响应对比
    plt.subplot(3, 1, 2)
    plt.plot(time, true_response, 'b-', linewidth=2, label='真实响应', alpha=0.8)
    plt.plot(time, predicted_response, 'r--', linewidth=2, label='预测响应', alpha=0.8)
    plt.ylabel('位移 (m)')
    plt.title('振动响应对比')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图3: 误差
    error = predicted_response - true_response
    plt.subplot(3, 1, 3)
    plt.plot(time, error, 'k-', linewidth=1.5, label='预测误差')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m)')
    plt.title('预测误差')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quick_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算误差统计
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    
    print("\n=== 预测结果统计 ===")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"最大误差: {np.max(np.abs(error)):.6f}")
    
    # 绘制训练损失
    plt.figure(figsize=(8, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== 演示完成 ===")
    print("结果图片已保存：")
    print("- quick_demo_results.png: 预测结果对比")
    print("- training_loss.png: 训练损失曲线")


if __name__ == "__main__":
    quick_demo()
