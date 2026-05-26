"""
PINN故障诊断系统配置文件
"""
import os

class Config:
    # 数据路径
    DATA_ROOT = r'C:\Users\ClearNight\Desktop\PINN\CWT_PCA_0HP'
    TRAIN_PATH = os.path.join(DATA_ROOT, 'train_data.mat')
    VAL_PATH = os.path.join(DATA_ROOT, 'val_data.mat')
    TEST_PATH = os.path.join(DATA_ROOT, 'test_data.mat')
    
    # 图像参数
    IMG_SIZE = 224
    IN_CHANNELS = 3      # 通道数: 3 = 幅值+cos(相位)+sin(相位); 1 = 仅幅值
    NUM_CLASSES = 10
    
    # 类别标签
    CLASS_NAMES = [
        'Normal',
        'IR007', 'IR014', 'IR021',
        'OR007', 'OR014', 'OR021',
        'Ball007', 'Ball014', 'Ball021'
    ]
    
    # 训练参数
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-3
    
    # PINN物理损失权重
    LAMBDA_PHYSICS = 0.01  # 物理约束: 频谱稀疏性，小权重避免过度抑制特征
    
    # 设备
    DEVICE = 'cuda'  # 'cuda' 或 'cpu'
    
    # ELM 分类器参数
    ELM_HIDDEN = 256        # ELM 隐藏节点数
    ELM_ACTIVATION = 'tanh' # 激活函数: 'tanh', 'sigmoid', 'relu'
    ELM_REG = 0.01          # 正则化系数
    
    # 分类器类型: 'softmax' (原始) 或 'elm' (两阶段)
    CLASSIFIER = 'softmax'
    
    # 卷积类型: 'fourier' (傅里叶卷积) 或 'standard' (普通Conv2d)
    CONV_TYPE = 'fourier'
    
    # 残差块类型: 'standard' (标准) 或 'depthwise' (深度可分离, 参数量减少88%)
    RESIDUAL_TYPE = 'depthwise'
    
    # 多头自注意力: True (开启) 或 False (关闭)
    USE_MHA = True
    
    # 保存路径
    SAVE_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    
    # 随机种子
    SEED = 42
