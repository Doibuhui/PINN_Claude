"""
数据加载模块
加载MATLAB预处理的CWT图像数据
支持MATLAB v7.3 (HDF5) 和旧版 .mat 格式
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os
import h5py


def _load_mat_file(mat_path):
    """加载 .mat 文件，自动兼容 v7.3 (HDF5) 和旧版格式"""
    try:
        data = loadmat(mat_path)
        # 检查是否真的读到了有效数据（loadmat 对 v7.3 可能返回空字典）
        if len([k for k in data if not k.startswith('__')]) > 0:
            return data, 'scipy'
    except NotImplementedError:
        pass

    # 使用 h5py 读取 v7.3 格式
    f = h5py.File(mat_path, 'r')
    data = {}
    for key in f.keys():
        if not key.startswith('#'):
            data[key] = np.array(f[key])
    f.close()
    return data, 'h5py'


class CWTDataset(Dataset):
    """CWT时频图数据集"""
    
    def __init__(self, mat_path, transform=None):
        """
        Args:
            mat_path: .mat文件路径
            transform: 数据增强变换
        """
        data, reader = _load_mat_file(mat_path)
        
        # 获取数据和标签
        if 'X_train' in data:
            self.images = data['X_train']
            self.labels = data['y_train'].flatten()
        elif 'X_val' in data:
            self.images = data['X_val']
            self.labels = data['y_val'].flatten()
        elif 'X_test' in data:
            self.images = data['X_test']
            self.labels = data['y_test'].flatten()
        else:
            raise ValueError(f"无法识别的数据格式: {mat_path}, keys: {list(data.keys())}")
        
        # h5py 读取的 v7.3 数据需要转置（MATLAB 列优先 → Python 行优先）
        if reader == 'h5py':
            self.images = np.transpose(self.images)
            self.labels = self.labels.flatten()
        
        # 转换为 [N, C, H, W] 格式
        if self.images.ndim == 3:
            # 灰度图: [H, W, N] → [N, H, W] → [N, 1, H, W]
            self.images = self.images[..., np.newaxis]  # [H, W, N, 1]
            self.images = np.transpose(self.images, (2, 3, 0, 1))  # [N, 1, H, W]
        elif self.images.ndim == 4:
            if reader == 'h5py':
                # h5py transpose后已是 [N, C, H, W]，无需再转
                pass
            else:
                # scipy加载: [H, W, C, N] → [N, C, H, W]
                self.images = np.transpose(self.images, (3, 2, 0, 1))
        
        # 确保是float32，并归一化到[0,1]
        if self.images.dtype == np.uint8:
            self.images = self.images.astype(np.float32) / 255.0
        else:
            self.images = self.images.astype(np.float32)
        self.labels = self.labels.astype(np.int64)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 先转为tensor，再做增强
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)


class GaussianNoise:
    """给张量添加高斯噪声"""
    def __init__(self, std=0.01, p=0.3):
        self.std = std
        self.p = p
    
    def __call__(self, x):
        if np.random.rand() < self.p:
            noise = torch.randn_like(x) * self.std
            x = x + noise
        return x


class TrainAugmentation:
    """CWT时频图数据增强 (纯PyTorch实现, 不依赖torchvision)"""
    def __init__(self, hflip_p=0.5, vflip_p=0.3, rotate_deg=5, noise_std=0.01, noise_p=0.3):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_deg = rotate_deg
        self.noise_std = noise_std
        self.noise_p = noise_p
    
    def __call__(self, x):
        # x: [C, H, W] tensor
        
        # 水平翻转 (时间轴)
        if np.random.rand() < self.hflip_p:
            x = x.flip(-1)
        
        # 垂直翻转 (频率轴)
        if np.random.rand() < self.vflip_p:
            x = x.flip(-2)
        
        # 小角度旋转 (用仿射矩阵实现)
        if self.rotate_deg > 0:
            angle = np.random.uniform(-self.rotate_deg, self.rotate_deg) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32)
            grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), x.unsqueeze(0).size(), align_corners=False)
            x = torch.nn.functional.grid_sample(x.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0)
        
        # 高斯噪声
        if np.random.rand() < self.noise_p:
            x = x + torch.randn_like(x) * self.noise_std
        
        return x


def get_dataloaders(config):
    """获取训练、验证、测试数据加载器"""
    
    train_dataset = CWTDataset(config.TRAIN_PATH, transform=TrainAugmentation())
    val_dataset = CWTDataset(config.VAL_PATH)
    test_dataset = CWTDataset(config.TEST_PATH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
