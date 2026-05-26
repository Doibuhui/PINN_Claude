"""
ELM 分类器模块
用于 PINN 两阶段训练的第二阶段：
冻结 PINN 特征提取网络，用 ELM 解析求解分类
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report


class ELMClassifier:
    """
    极限学习机分类器
    
    原理:
        1. 随机输入权重 W_in 和偏置 b (不训练)
        2. 隐藏层输出 H = activation(X @ W_in + b)
        3. 输出权重解析求解: β = (H^T H + λI)^{-1} H^T Y
        4. 预测: Y_pred = H @ β
    
    优势:
        - 训练极快 (无迭代, 一次矩阵运算)
        - 不会过拟合隐藏层权重
        - 适合与 PINN 特征提取器配合使用
    """
    
    def __init__(self, n_hidden=256, activation='tanh', reg=0.01, random_state=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.reg = reg
        self.random_state = random_state
        
        self.input_weights = None
        self.bias = None
        self.output_weights = None
        self.classes_ = None
        
    def _activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return np.tanh(x)
    
    def fit(self, X, y):
        """
        解析求解输出权重
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            y: 标签 [n_samples]
        """
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # one-hot 编码
        y_onehot = np.zeros((n_samples, n_classes))
        for i, cls in enumerate(self.classes_):
            y_onehot[y == cls, i] = 1
        
        # 随机输入权重和偏置 (Xavier 初始化)
        limit = np.sqrt(6.0 / (n_features + self.n_hidden))
        self.input_weights = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.bias = np.random.uniform(-limit, limit, (1, self.n_hidden))
        
        # 隐藏层输出
        H = self._activate(X @ self.input_weights + self.bias)
        
        # 解析求解输出权重: β = (H^T H + λI)^{-1} H^T Y
        if n_samples > self.n_hidden:
            # 样本数 > 隐藏节点数: 使用 (H^T H + λI)^{-1}
            A = H.T @ H + self.reg * np.eye(self.n_hidden)
            self.output_weights = np.linalg.solve(A, H.T @ y_onehot)
        else:
            # 样本数 < 隐藏节点数: 使用 (I + λ H H^T)^{-1} Y (Woodbury 恒等式)
            A = np.eye(n_samples) + self.reg * (H @ H.T)
            self.output_weights = H.T @ np.linalg.solve(A, y_onehot)
        
        return self
    
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        H = self._activate(X @ self.input_weights + self.bias)
        scores = H @ self.output_weights
        return np.array([self.classes_[idx] for idx in np.argmax(scores, axis=1)])
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        H = self._activate(X @ self.input_weights + self.bias)
        scores = H @ self.output_weights
        # softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class ELMWithPINN:
    """
    PINN + ELM 两阶段分类器
    
    阶段一: 训练 PINN 特征提取器 (FourierConv + ResNet)
    阶段二: 冻结 PINN, 提取特征, 用 ELM 解析求解分类
    """
    
    def __init__(self, model, config, n_hidden=256, activation='tanh', reg=0.01):
        self.model = model
        self.config = config
        self.elm = ELMClassifier(
            n_hidden=n_hidden,
            activation=activation,
            reg=reg,
            random_state=config.SEED
        )
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
    def extract_features(self, dataloader):
        """
        用训练好的 PINN 提取所有样本的 256 维特征
        
        Returns:
            features: [N, 256] numpy 数组
            labels: [N] numpy 数组
        """
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features, _ = self.model.extract_features(images)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        return np.concatenate(all_features), np.concatenate(all_labels)
    
    def fit_elm(self, train_loader, val_loader=None):
        """
        阶段二: 提取特征 + ELM 训练
        """
        print("\n" + "=" * 60)
        print("阶段二: ELM 分类器训练")
        print("=" * 60)
        
        # 提取训练集特征
        print("提取训练集特征...")
        train_features, train_labels = self.extract_features(train_loader)
        print(f"  训练特征: {train_features.shape}, 标签: {train_labels.shape}")
        
        # ELM 训练 (解析求解, 瞬间完成)
        print("ELM 解析求解输出权重...")
        self.elm.fit(train_features, train_labels)
        print(f"  隐藏节点数: {self.elm.n_hidden}")
        print(f"  输出权重形状: {self.elm.output_weights.shape}")
        
        # 训练集评估
        train_pred = self.elm.predict(train_features)
        train_acc = accuracy_score(train_labels, train_pred)
        print(f"  训练集准确率: {train_acc:.4f} ({train_acc * 100:.2f}%)")
        
        # 验证集评估
        if val_loader is not None:
            val_features, val_labels = self.extract_features(val_loader)
            val_pred = self.elm.predict(val_features)
            val_acc = accuracy_score(val_labels, val_pred)
            print(f"  验证集准确率: {val_acc:.4f} ({val_acc * 100:.2f}%)")
        
        return self
    
    def evaluate(self, test_loader, class_names=None):
        """
        在测试集上评估
        """
        print("\n" + "=" * 60)
        print("ELM 测试集评估")
        print("=" * 60)
        
        test_features, test_labels = self.extract_features(test_loader)
        test_pred = self.elm.predict(test_features)
        
        accuracy = accuracy_score(test_labels, test_pred)
        f1 = f1_score(test_labels, test_pred, average='weighted')
        
        print(f"测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"F1 分数 (加权): {f1:.4f} ({f1 * 100:.2f}%)")
        
        if class_names is not None:
            report = classification_report(
                test_labels, test_pred,
                target_names=class_names,
                digits=4
            )
            print(f"\n分类报告:\n{report}")
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'predictions': test_pred,
            'labels': test_labels
        }
