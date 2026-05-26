"""
评估模块
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os


class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, config, test_loader):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_best_model(self):
        """加载最佳模型"""
        checkpoint_path = os.path.join(self.config.SAVE_DIR, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载最佳模型，验证准确率: {checkpoint['val_acc']:.2f}%")
        else:
            print("未找到最佳模型，使用当前模型")
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in self.test_loader:
            images = images.to(self.device)
            
            logits, _ = self.model(images)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算准确率
        accuracy = np.mean(all_preds == all_labels) * 100
        
        # 分类报告
        report = classification_report(
            all_labels, all_preds,
            target_names=self.config.CLASS_NAMES,
            digits=4
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'report': report,
            'confusion_matrix': cm
        }
    
    def print_results(self, results):
        """打印评估结果"""
        total_params = sum(p.numel() for p in self.model.parameters())
        print("\n" + "=" * 60)
        print("测试集评估结果")
        print("=" * 60)
        print(f"模型参数量: {total_params:,}")
        print(f"总体准确率: {results['accuracy']:.2f}%")
        print("\n分类报告:")
        print(results['report'])
        print("=" * 60)
