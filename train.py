"""
训练模块
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time


class Trainer:
    """PINN训练器"""
    
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS
        )
        
        # 设备
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 创建保存目录
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        
        # 记录最佳验证准确率
        self.best_val_acc = 0.0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS}')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            logits, physics_features = self.model(images)
            
            # 计算损失
            cls_loss = self.criterion(logits, labels)
            physics_loss = self.model.physics_loss(physics_features)
            
            # 总损失 = 分类损失 + 物理约束损失
            loss = cls_loss + self.config.LAMBDA_PHYSICS * physics_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """完整训练流程"""
        print(f"开始训练，设备: {self.device}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
        print("-" * 60)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(self.config.EPOCHS):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{self.config.EPOCHS}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_path = os.path.join(self.config.SAVE_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f'  [OK] best model saved, val_acc: {val_acc:.2f}%')
            
            print("-" * 60)
        
        print(f"训练完成！最佳验证准确率: {self.best_val_acc:.2f}%")
        
        return history
