"""
数据加载模块
支持两种模式:
  - CWTDataset: 加载MATLAB预计算的CWT图像 (USE_RAW_SIGNAL=False)
  - CWRURawDataset: 从原始CWRU .mat文件加载振动信号 (USE_RAW_SIGNAL=True, 端到端LearnableCWT)
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

    if getattr(config, 'USE_RAW_SIGNAL', False):
        # 端到端模式: 从原始信号加载, 模型内部做 LearnableCWT
        train_dataset = CWRURawDataset(config.RAW_DATA_DIR, 'train',
                                       use_med=getattr(config, 'USE_MED', False),
                                       seed=config.SEED)
        val_dataset   = CWRURawDataset(config.RAW_DATA_DIR, 'val',
                                       use_med=getattr(config, 'USE_MED', False),
                                       seed=config.SEED)
        test_dataset  = CWRURawDataset(config.RAW_DATA_DIR, 'test',
                                       use_med=getattr(config, 'USE_MED', False),
                                       seed=config.SEED)
    else:
        # 传统模式: 加载 MATLAB 预计算的 CWT 图像
        train_dataset = CWTDataset(config.TRAIN_PATH, transform=TrainAugmentation())
        val_dataset   = CWTDataset(config.VAL_PATH)
        test_dataset  = CWTDataset(config.TEST_PATH)
    
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


# ================================================================
# CWRU 原始振动信号数据集 (USE_RAW_SIGNAL=True 时使用)
# ================================================================

class CWRURawDataset(Dataset):
    """
    直接从 CWRU 原始 .mat 文件加载 DE_time 振动信号
    滑窗切分, 按 load+class 独立划分 train/val/test (时间顺序, 避免泄露)
    """

    LABEL_MAP = [
        ['97',  '98',  '99',  0, 'Normal'],
        ['105', '106', '107', 1, 'IR007'],
        ['169', '170', '171', 2, 'IR014'],
        ['209', '210', '211', 3, 'IR021'],
        ['130', '131', '132', 4, 'OR007'],
        ['197', '198', '199', 5, 'OR014'],
        ['234', '235', '236', 6, 'OR021'],
        ['118', '119', '120', 7, 'Ball007'],
        ['185', '186', '187', 8, 'Ball014'],
        ['222', '223', '224', 9, 'Ball021'],
    ]

    def __init__(self, raw_data_dir, split='train',
                 window_len=1024, step=512, sampling_rate=12000,
                 train_ratio=0.85, use_med=False,
                 med_filter_size=40, med_max_iter=30,
                 seed=42):
        self.window_len = window_len
        self.sampling_rate = sampling_rate

        load_dirs = {
            'CWRU0HP': os.path.join(raw_data_dir, 'CWRU0HP'),
            'CWRU1HP': os.path.join(raw_data_dir, 'CWRU1HP'),
            'CWRU2HP': os.path.join(raw_data_dir, 'CWRU2HP'),
        }

        rng = np.random.RandomState(seed)

        all_signals = []
        all_labels = []
        all_loads = []

        for load_name, load_path in load_dirs.items():
            load_idx = int(load_name.replace('CWRU', '').replace('HP', ''))
            for row in self.LABEL_MAP:
                fname = row[load_idx] + '.mat'
                fpath = os.path.join(load_path, fname)
                if not os.path.exists(fpath):
                    continue
                signal = self._load_signal(fpath, fname)
                if signal is None:
                    continue

                if use_med:
                    signal = med_deconv(signal, med_filter_size, med_max_iter)

                cls = row[3]
                num_samples = (len(signal) - window_len) // step + 1
                for i in range(num_samples):
                    seg = signal[i * step : i * step + window_len]
                    all_signals.append(seg)
                    all_labels.append(cls)
                    all_loads.append(load_name)

        all_signals = np.array(all_signals, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        all_loads = np.array(all_loads)

        # 按 load+class 独立打乱
        train_loads = {'CWRU0HP', 'CWRU1HP'}   # 训练/验证来源
        test_loads  = {'CWRU2HP'}               # 测试来源 (跨负载泛化)

        train_idx, val_idx, test_idx = [], [], []
        for load_name in load_dirs:
            for cls in range(10):
                mask = (all_loads == load_name) & (all_labels == cls)
                idx = np.where(mask)[0]
                rng.shuffle(idx)
                if load_name in train_loads:
                    n_tr = int(len(idx) * train_ratio)
                    train_idx.append(idx[:n_tr])
                    val_idx.append(idx[n_tr:])
                else:
                    test_idx.append(idx)

        if split == 'train':
            sel = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
        elif split == 'val':
            sel = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
        else:
            sel = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)

        rng.shuffle(sel)
        self.signals = all_signals[sel]
        self.labels = all_labels[sel]

    def _load_signal(self, fpath, fname):
        """精确匹配 DE_time 变量名 (避免 99.mat 含 X098+X099 的问题)"""
        from scipy.io import loadmat
        try:
            data = loadmat(fpath)
        except NotImplementedError:
            import h5py
            with h5py.File(fpath, 'r') as f:
                data = {k: np.array(v) for k, v in f.items() if not k.startswith('#')}

        file_id = fname.replace('.mat', '')
        expected = 'X' + file_id.zfill(3) + '_DE_time'
        for k in data:
            if k == expected:
                return data[k].flatten()
        # fallback
        for k in data:
            if 'DE' in k and not k.startswith('__'):
                return data[k].flatten()
        return None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = torch.from_numpy(self.signals[idx]).unsqueeze(0)  # [1, window_len]
        # 幅值归一化
        sig = sig / (sig.abs().max() + 1e-8)
        return sig, torch.tensor(self.labels[idx])


def med_deconv(x, filter_size=40, max_iter=30, tol=1e-6):
    """
    最小熵解卷积 (MED): 增强冲击特征, 剥离传递路径影响
    Python 实现 (Wiggins 1978), 等价于 MATLAB 版
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    N = len(x)

    pad = filter_size // 2
    x_pad = np.concatenate([x[:pad][::-1], x, x[-pad:][::-1]])

    X = np.zeros((N, filter_size))
    for i in range(N):
        X[i, :] = x_pad[i:i + filter_size][::-1]

    f = np.zeros(filter_size)
    f[filter_size // 2] = 1.0

    for _ in range(max_iter):
        y = X @ f
        Ey2 = np.mean(y ** 2)
        Ey4 = np.mean(y ** 4)
        if Ey2 < np.finfo(float).eps or Ey4 < np.finfo(float).eps:
            break

        b = (Ey2 / Ey4) * y ** 3 - y
        Rxx = X.T @ X / N
        rxb = X.T @ b / N

        lam = 0.01 * np.trace(Rxx) / filter_size
        f_new = np.linalg.solve(Rxx + lam * np.eye(filter_size), rxb)

        f_norm = np.linalg.norm(f_new)
        if f_norm < np.finfo(float).eps:
            break
        f_new /= f_norm

        if np.linalg.norm(f_new - f) / f_norm < tol:
            f = f_new
            break
        f = f_new

    return (X @ f).astype(np.float32)
