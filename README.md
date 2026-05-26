# PINN故障诊断系统

基于物理信息神经网络（Physics-Informed Neural Network）的轴承故障诊断系统，使用CWRU（西储大学）轴承数据集。

## 项目结构

```
PINN/
├── config.py           # 配置文件（路径、参数等）
├── dataloader.py       # 数据加载模块
├── pinn_model.py       # PINN模型定义
├── train.py            # 训练模块
├── evaluate.py         # 评估模块
├── utils.py            # 工具函数
├── main.py             # 主入口
├── requirements.txt    # Python依赖包
└── README.md           # 说明文档
```

## 环境配置

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.2.0

### 2. 确认数据文件

确保MATLAB预处理生成的数据文件位于正确路径：

```
C:\Users\ClearNight\Desktop\FCELM\CWT_Matrices\
├── train_data.mat
├── val_data.mat
└── test_data.mat
```

如路径不同，请修改 `config.py` 中的 `DATA_ROOT` 变量。

## 使用方法

### 1. 直接运行完整流程

```bash
python main.py
```

这将依次执行：数据加载 → 模型训练 → 模型评估 → 保存结果

### 2. 自定义训练参数

编辑 `config.py` 文件：

```python
class Config:
    # 修改训练轮数
    EPOCHS = 50
    
    # 修改批大小
    BATCH_SIZE = 16
    
    # 修改学习率
    LEARNING_RATE = 5e-4
    
    # 修改物理损失权重
    LAMBDA_PHYSICS = 0.05
    
    # 使用CPU训练（无GPU时）
    DEVICE = 'cpu'
```

### 3. 单独调用各模块

```python
from config import Config
from dataloader import get_dataloaders
from pinn_model import PINNFaultDiagnosis
from train import Trainer
from evaluate import Evaluator

# 加载配置
config = Config()

# 加载数据
train_loader, val_loader, test_loader = get_dataloaders(config)

# 创建模型
model = PINNFaultDiagnosis(config)

# 训练
trainer = Trainer(model, config, train_loader, val_loader)
history = trainer.train()

# 评估
evaluator = Evaluator(model, config, test_loader)
evaluator.load_best_model()
results = evaluator.evaluate()
```

## 模型架构

### PINN（物理信息神经网络）

```
输入 (1×224×224)
    ↓
物理约束块 (提取频域特征)
    ↓
残差网络层1 (32通道)
    ↓
残差网络层2 (64通道)
    ↓
残差网络层3 (128通道)
    ↓
残差网络层4 (256通道)
    ↓
全局平均池化
    ↓
全连接层 (10类输出)
```

### 损失函数

```
总损失 = 分类损失 + λ × 物理约束损失
```

- 分类损失：交叉熵损失
- 物理约束损失：频域平滑性约束
- λ：物理损失权重（默认0.1）

## 输出结果

训练完成后，将在以下目录生成结果：

```
checkpoints/
└── best_model.pth      # 最佳模型权重

results/
├── confusion_matrix.png    # 混淆矩阵
└── training_history.png    # 训练曲线
```

## 故障类别

| 标签 | 类别 | 描述 |
|------|------|------|
| 0 | Normal | 正常状态 |
| 1 | IR007 | 内圈故障 0.007英寸 |
| 2 | IR014 | 内圈故障 0.014英寸 |
| 3 | IR021 | 内圈故障 0.021英寸 |
| 4 | OR007 | 外圈故障 0.007英寸 |
| 5 | OR014 | 外圈故障 0.014英寸 |
| 6 | OR021 | 外圈故障 0.021英寸 |
| 7 | Ball007 | 滚动体故障 0.007英寸 |
| 8 | Ball014 | 滚动体故障 0.014英寸 |
| 9 | Ball021 | 滚动体故障 0.021英寸 |

## 常见问题

### Q1: CUDA内存不足

减小批大小：
```python
# config.py
BATCH_SIZE = 8  # 或更小
```

### Q2: 数据路径错误

检查并修改 `config.py` 中的路径：
```python
DATA_ROOT = r'你的数据路径'
```

### Q3: 训练速度慢

- 确保安装了GPU版本的PyTorch
- 减小图像尺寸或模型复杂度
- 减少训练轮数

## 引用

如果使用了CWRU数据集，请引用：
```
Case Western Reserve University Bearing Data Center.
https://engineering.case.edu/bearingdatacenter
```

## 许可证

本项目仅供学术研究使用。
