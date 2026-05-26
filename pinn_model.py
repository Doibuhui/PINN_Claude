"""
PINN模型 - 物理信息神经网络
用于轴承故障诊断
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineActivation(nn.Module):
    """
    B-Spline 可学习激活函数 (KAN 核心组件) - 快速实现
    
    原理:
        y = Σ c_i · B_i(x)
        B_i 是 B-spline 基函数, c_i 是可学习系数。
    
    快速实现:
        1. 初始化时在均匀网格上预计算所有基函数值
        2. 运行时通过线性插值查表求值 (O(1) per point, 无递归)
        3. 完全可微分, 支持反向传播
    """
    
    def __init__(self, num_splines, num_control_points=8, spline_order=3,
                 grid_size=100, x_range=(-1.5, 1.5)):
        super().__init__()
        self.num_cp = num_control_points
        self.order = spline_order
        self.grid_size = grid_size
        self.x_range = x_range
        self.num_knots = num_control_points + spline_order + 1
        
        # 可学习控制点系数
        c_init = torch.linspace(-1, 1, num_control_points).unsqueeze(0).expand(num_splines, -1)
        self.coeffs = nn.Parameter(c_init.clone())
        
        # 预计算基函数矩阵 (固定, 不训练)
        # 在均匀网格上计算 B-spline 基函数值
        knots = torch.linspace(x_range[0], x_range[1], self.num_knots)
        grid = torch.linspace(x_range[0], x_range[1], grid_size)
        
        # 0阶基函数
        bases = torch.zeros(num_control_points, grid_size)
        for i in range(num_control_points):
            bases[i] = ((grid >= knots[i]) & (grid < knots[i + 1])).float()
        
        # 递推升阶
        for k in range(1, spline_order + 1):
            new_bases = torch.zeros_like(bases)
            for i in range(num_control_points):
                d1 = knots[i + k] - knots[i]
                d2 = knots[i + k + 1] - knots[i + 1]
                if d1 > 1e-8:
                    new_bases[i] += (grid - knots[i]) / d1 * bases[i]
                if d2 > 1e-8 and (i + 1) < num_control_points:
                    new_bases[i] += (knots[i + k + 1] - grid) / d2 * bases[i + 1]
            bases = new_bases
        
        # basis_table: [num_cp, grid_size] - 预计算的基函数查表
        self.register_buffer('basis_table', bases)
        self.register_buffer('grid', grid)
        
    def forward(self, x):
        """
        x: [*, num_splines] 最后一维是 spline 索引
        返回: [*, num_splines]
        """
        x_norm = torch.tanh(x)  # 归一化到 [-1, 1]
        orig_shape = x_norm.shape
        num_splines = orig_shape[-1]
        
        # 将 x 映射到网格索引
        x_clamped = x_norm.clamp(self.x_range[0] + 1e-6, self.x_range[1] - 1e-6)
        # 归一化到 [0, grid_size-1]
        idx_float = (x_clamped - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.grid_size - 1)
        idx_floor = idx_float.long().clamp(0, self.grid_size - 2)
        idx_frac = idx_float - idx_floor.float()  # 小数部分, 用于插值
        
        # 从查表中取基函数值并线性插值
        # basis_table: [num_cp, grid_size]
        # idx_floor: [*, num_splines]
        left_vals = self.basis_table[:, idx_floor]   # [num_cp, *, num_splines]
        right_vals = self.basis_table[:, idx_floor + 1]
        
        # 插值: basis = left + frac * (right - left)
        frac = idx_frac.unsqueeze(0)  # [1, *, num_splines]
        basis_vals = left_vals + frac * (right_vals - left_vals)  # [num_cp, *, num_splines]
        
        # spline 求值: y_s = Σ c_{s,i} · B_i(x_s)
        # basis_vals: [num_cp, *, S] -> [*, S, num_cp]
        basis_vals = basis_vals.permute(*range(1, len(orig_shape) + 1), 0)
        # coeffs: [S, num_cp]
        y = (basis_vals * self.coeffs).sum(dim=-1)  # [*, S]
        
        return y


class FourierConv2d(nn.Module):
    """
    KAN-增强傅里叶卷积层
    
    与标准傅里叶卷积的区别:
        标准: out_fft = W · x_fft        (线性权重乘法)
        KAN:  out_fft = φ_real(x_fft) + j·φ_imag(x_fft)
              其中 φ 是 B-spline 可学习函数
    
    这使得频域滤波不再是简单的线性缩放，
    而是可以学习任意非线性频率响应函数。
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 num_control_points=8, spline_order=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 每个输入通道一条独立的 B-spline (实部/虚部各一条)
        # 批量化: 一次调用处理所有通道
        self.spline_real = BSplineActivation(in_channels, num_control_points, spline_order)
        self.spline_imag = BSplineActivation(in_channels, num_control_points, spline_order)
        
        # 线性混合系数
        self.mix_real = nn.Parameter(torch.randn(out_channels, in_channels) * 0.02)
        self.mix_imag = nn.Parameter(torch.randn(out_channels, in_channels) * 0.02)
        
        # 局部空间卷积
        self.local_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding=padding, bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_real = x_fft.real  # [B, C, H, W']
        x_imag = x_fft.imag  # [B, C, H, W']
        
        # 2. 批量 KAN spline 变换: [B, C, H, W'] -> [B, C, H, W']
        # 重塑为 [B*H'*W', C] 以便 spline 按最后一维处理各通道
        Wp = x_real.shape[-1]
        sr = self.spline_real(x_real.permute(0, 2, 3, 1).reshape(-1, C))  # [B*H'*W', C]
        si = self.spline_imag(x_imag.permute(0, 2, 3, 1).reshape(-1, C))
        sr = sr.reshape(B, H, Wp, C).permute(0, 3, 1, 2)  # [B, C, H, W']
        si = si.reshape(B, H, Wp, C).permute(0, 3, 1, 2)
        
        # 3. 线性混聚合到输出通道
        mr = self.mix_real.view(1, self.out_channels, self.in_channels, 1, 1)
        mi = self.mix_imag.view(1, self.out_channels, self.in_channels, 1, 1)
        tr = sr.unsqueeze(1)  # [B, 1, C, H, W']
        ti = si.unsqueeze(1)
        
        out_real = (mr * tr - mi * ti).sum(dim=2)
        out_imag = (mr * ti + mi * tr).sum(dim=2)
        
        # 4. IFFT
        out_fft = torch.complex(out_real, out_imag)
        x_spectral = torch.fft.irfft2(out_fft, s=(H, W), norm='ortho')
        
        # 5. 局部卷积 + 融合
        x_local = self.local_conv(x_spectral)
        out = x_spectral + x_local
        out = F.relu(self.bn(out))
        
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    将特征向量拆分为多个 token，学习 token 之间的相互关系
    """
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class PhysicsBlock(nn.Module):
    """物理约束块：提取频率特征"""
    
    def __init__(self, in_channels, conv_type='fourier'):
        super().__init__()
        if conv_type == 'fourier':
            self.conv1 = FourierConv2d(in_channels, 16)
            self.conv2 = FourierConv2d(16, 16)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    """标准残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积: Depthwise(空间) + Pointwise(通道)"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        return self.pw(self.dw(x))


class DepthwiseSeparableResidualBlock(nn.Module):
    """深度可分离残差块: 残差拓扑不变，卷积换成深度可分离"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class PINNFaultDiagnosis(nn.Module):
    """
    物理信息神经网络故障诊断模型
    结合CNN特征提取和物理约束
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 物理约束块
        conv_type = getattr(config, 'CONV_TYPE', 'fourier')
        self.physics_block = PhysicsBlock(config.IN_CHANNELS, conv_type=conv_type)
        
        # CNN特征提取
        self.layer1 = self._make_layer(16, 32, 2, stride=2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 多头自注意力 (可选)
        self.use_mha = getattr(config, 'USE_MHA', True)
        if self.use_mha:
            self.mha = MultiHeadSelfAttention(dim=256, num_heads=8, dropout=0.1)
            self.mha_norm = nn.LayerNorm(256)
        
        # 分类器
        self.fc = nn.Linear(256, config.NUM_CLASSES)
        
        # 物理约束参数（可学习）
        self.freq_weight = nn.Parameter(torch.ones(1, 16, 1, 1))
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        residual_type = getattr(self.config, 'RESIDUAL_TYPE', 'standard')
        block = DepthwiseSeparableResidualBlock if residual_type == 'depthwise' else ResidualBlock
        
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def physics_loss(self, features):
        """
        物理约束: 频谱稀疏性

        轴承故障信号 = 周期性冲击脉冲
        → 调制谱能量集中在故障特征频率及其倍频
        → 频谱应呈现稀疏分布（少数频率分量集中大部分能量）

        L1/L2 稀疏度量:
          值 ≈ 1    → 完美稀疏 (单频峰, 理想故障信号)
          值 ≈ √N   → 完全平坦 (白噪声)
          值越小     → 频谱越稀疏 → 越符合故障物理特征

        与 FourierConv 的协同:
          FourierConv 学习频域非线性变换用于判别
          稀疏性 loss 约束特征频谱呈现"稀疏峰值"形态
          两者方向一致: 找到并保留尖锐的频域特征
        """
        # 沿时间轴做 1D FFT → 调制谱 (包络谱近似)
        time_spectrum = torch.fft.rfft(features, dim=-1)
        time_mag = torch.abs(time_spectrum)  # [B, C, H, W_freq]

        eps = 1e-8
        # 空间维度和频率维度取均值，保留 batch 和 channel
        l1 = time_mag.mean(dim=[2, 3])
        l2 = torch.sqrt((time_mag ** 2).mean(dim=[2, 3]) + eps)
        sparsity = (l1 / (l2 + eps)).mean()  # 标量

        return sparsity
    
    def extract_features(self, x):
        physics_features = self.physics_block(x)
        physics_features = physics_features * self.freq_weight
        x = self.layer1(physics_features)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.use_mha:
            B, C, H, W = x.shape
            x_tokens = x.flatten(2).transpose(1, 2)
            x_attn = self.mha(x_tokens)
            x_attn = x_tokens + x_attn
            x_attn = self.mha_norm(x_attn)
            x = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x, physics_features
    
    def forward(self, x):
        physics_features = self.physics_block(x)
        physics_features = physics_features * self.freq_weight
        
        x = self.layer1(physics_features)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.use_mha:
            B, C, H, W = x.shape
            x_tokens = x.flatten(2).transpose(1, 2)
            x_attn = self.mha(x_tokens)
            x_attn = x_tokens + x_attn
            x_attn = self.mha_norm(x_attn)
            x = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        
        return logits, physics_features
