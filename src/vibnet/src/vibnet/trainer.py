"""
模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict
import time


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def load_best_model(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_epoch(model: nn.Module, train_loader: DataLoader, 
               optimizer: optim.Optimizer, criterion: nn.Module,
               device: torch.device) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch(model: nn.Module, val_loader: DataLoader,
                  criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_mae = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 计算MAE
            mae = torch.mean(torch.abs(outputs - targets))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'mae': total_mae / num_batches
    }


def train_model(model: nn.Module, train_data: DataLoader, val_data: DataLoader,
               device: torch.device, epochs: int = 100, batch_size: int = 32,
               learning_rate: float = 0.001, patience: int = 15) -> nn.Module:
    """
    训练模型
    
    Args:
        model: 待训练的模型
        train_data: 训练数据加载器
        val_data: 验证数据加载器
        device: 计算设备
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        patience: 早停耐心值
        
    Returns:
        训练好的模型
    """
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
      # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=patience)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'lr': []
    }
    
    print(f"开始训练，共 {epochs} 个epoch")
    print(f"设备: {device}")
    print(f"初始学习率: {learning_rate}")
    print("-" * 60)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_data, optimizer, criterion, device)
        
        # 验证
        val_metrics = validate_epoch(model, val_data, criterion, device)
        val_loss = val_metrics['loss']
        val_mae = val_metrics['mae']
        
        # 学习率调整
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val MAE: {val_mae:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f"\n早停触发，在第 {epoch+1} epoch停止训练")
            break
    
    # 加载最佳模型
    early_stopping.load_best_model(model)
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"训练完成！总用时: {total_time:.2f}s")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 保存训练历史
    np.save('training_history.npy', history)
    
    return model


def plot_training_history(history_path: str = 'training_history.npy'):
    """绘制训练历史"""
    try:
        import matplotlib.pyplot as plt
        
        history = np.load(history_path, allow_pickle=True).item()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('训练历史', fontsize=16)
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label='训练损失', color='blue')
        axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE曲线
        axes[0, 1].plot(history['val_mae'], label='验证MAE', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('平均绝对误差')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[1, 0].plot(history['lr'], label='学习率', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 验证损失放大图
        axes[1, 1].plot(history['val_loss'], label='验证损失', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title('验证损失 (放大)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练历史图已保存为 training_history.png")
        
    except ImportError:
        print("matplotlib未安装，无法绘制训练历史图")
    except Exception as e:
        print(f"绘制训练历史图时出错: {e}")


if __name__ == "__main__":
    # 如果直接运行此文件，绘制训练历史
    plot_training_history()
