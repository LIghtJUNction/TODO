import torch
import numpy as np
from pathlib import Path
import os
import sys

# 添加当前目录到路径，以便直接运行时能找到模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .model import VibrationNet
    from .data_generator import generate_sdof_data
    from .trainer import train_model
    from .predictor import predict_response, plot_results
except ImportError:
    # 直接运行时使用绝对导入
    from model import VibrationNet
    from data_generator import generate_sdof_data
    from trainer import train_model
    from predictor import predict_response, plot_results
    
def main():
    """主程序：训练和测试单自由度系统振动响应预测模型"""
    print("=== 单自由度系统振动响应预测 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成训练数据
    print("\n1. 生成训练数据...")
    train_data, val_data = generate_sdof_data(
        n_samples=100,  # 减少样本数量以便快速测试
        val_split=0.2,
        sequence_length=50,  # 减少序列长度
        dt=0.01
    )
    
    # 创建模型
    print("\n2. 创建神经网络模型...")
    model = VibrationNet(
        input_size=12,  # 系统参数(3) + 激励力特征(9)
        hidden_sizes=[64, 128, 64],
        output_size=50,  # 位移响应序列长度（匹配序列长度）
        dropout_rate=0.1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练模型
    print("\n3. 开始训练模型...")
    trained_model = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=device,
        epochs=20,  # 减少训练轮数以便快速测试
        batch_size=16,
        learning_rate=0.001
    )
    
    # 保存模型
    model_path = Path("vibration_model.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 测试预测
    print("\n4. 测试模型预测...")
    # 创建测试样本
    test_mass = 1.0
    test_damping = 0.05  
    test_force = np.sin(np.linspace(0, 5, 50)) * 100  # 正弦激励力（匹配序列长度）
    
    predicted_response = predict_response(
        model=trained_model,
        mass=test_mass,
        damping_ratio=test_damping,
        force=test_force,
        device=device
    )
    
    # 生成真实响应用于对比
    try:
        from .data_generator import sdof_response
    except ImportError:
        from data_generator import sdof_response
    
    true_response = sdof_response(
        force=test_force,
        mass=test_mass,
        damping_ratio=test_damping,
        dt=0.01
    )
    
    # 绘制结果
    plot_results(
        time=np.linspace(0, 0.5, 50),  # 匹配序列长度
        true_response=true_response,
        predicted_response=predicted_response,
        force=test_force
    )
    
    print("\n=== 程序完成 ===")


if __name__ == "__main__":
    main()
