% dataprocessing_YSU_V_0HP.m
% YSU_V 单负载 0HP: 多文件拼接 → 滑窗 → CWT → PCA (8→3) → RGB图 → 划分
% 4类: Ball(0), Inner(1), Outer(2), Normal(3)
% 每类多个 .mat 文件拼接为长信号, 8通道, 滑动窗口512, 重叠50%

clear; clc; close all;

%% ========== 参数设置 ==========
window_len    = 512;
overlap       = 256;            % 50% overlap
step          = window_len - overlap;
sampling_rate = 20000;          % 采样率 20KHz
wavelet       = 'amor';
voices        = 48;
target_size   = [224 224];      % CWT图像目标尺寸
train_ratio   = 0.7;
val_ratio     = 0.15;
test_ratio    = 0.15;

%% ========== 路径配置 ==========
data_dir    = 'C:\Users\ClearNight\Desktop\YSU_V\0HP';
output_root = 'C:\Users\ClearNight\Desktop\PINN\CWT_PCA_0HP';
mat_dir     = fullfile(output_root, 'ClassMatrices');

%% ========== 类别目录映射 ==========
% 子目录按字母排序后顺序: 内圈, 外圈, 正常, 滚动体
% 映射到标签: Ball=0, Inner=1, Outer=2, Normal=3
temp_subdirs = dir(data_dir);
subdir_list  = temp_subdirs([temp_subdirs.isdir] & ~ismember({temp_subdirs.name}, {'.', '..'}));
subdir_names = {subdir_list.name};  % {'内圈', '外圈', '正常', '滚动体'}

% subdir index → class label
class_from_subdir = [1, 2, 3, 0];  % 内圈→1, 外圈→2, 正常→3, 滚动体→0

NUM_CLASSES = 4;

%% ========== 1. 拼接 + 滑窗 + CWT + PCA → RGB ==========
fprintf('========== 开始生成 RGB CWT 图像 (0HP) ==========\n');

for class = 0:3
    mkdir(fullfile(mat_dir, ['class_', num2str(class)]));
end

total_generated = 0;

for sd = 1:length(subdir_names)
    subdir_name  = subdir_names{sd};
    class_label  = class_from_subdir(sd);
    subdir_path  = fullfile(data_dir, subdir_name);
    
    % 获取该子目录下所有 .mat 文件
    mat_files = dir(fullfile(subdir_path, '*.mat'));
    if isempty(mat_files)
        warning('子目录 %s 中无 .mat 文件，跳过', subdir_name);
        continue;
    end
    
    % 拼接到一起
    all_data = cell(1, length(mat_files));
    for f = 1:length(mat_files)
        loaded = load(fullfile(subdir_path, mat_files(f).name));
        seg = loaded.data;  % [N, 8]
        if size(seg, 1) == 8 && size(seg, 2) ~= 8
            seg = seg';
        end
        all_data{f} = seg;
    end
    signal = cat(1, all_data{:});  % [total_N, 8]
    
    [sig_len, n_ch] = size(signal);
    fprintf('[类别 %d] %s: 总信号长度=%d, 通道数=%d, 文件数=%d\n', ...
            class_label, subdir_name, sig_len, n_ch, length(mat_files));
    
    % 滑动窗口切片
    num_samples = floor((sig_len - window_len) / step) + 1;
    fprintf('  可生成 %d 个样本\n', num_samples);
    
    for s = 1:num_samples
        start_idx = (s-1) * step + 1;
        seg = signal(start_idx : start_idx + window_len - 1, :);  % [512, 8]
        
        % 对8通道分别做CWT
        cwt_maps = zeros([target_size, n_ch]);  % [224, 224, 8]
        for ch = 1:n_ch
            [cfs, ~] = cwt(seg(:, ch), wavelet, sampling_rate, 'VoicesPerOctave', voices);
            mag = abs(cfs);
            mag_resized = imresize(mag, target_size);
            cwt_maps(:, :, ch) = mag_resized;
        end
        
        % PCA: 8通道 → 3通道
        pixels = reshape(cwt_maps, [], n_ch);  % [224*224, 8]
        [coeff, score, ~] = pca(pixels);
        rgb_pixels = score(:, 1:3);  % [224*224, 3]
        
        % 归一化到 [0, 255] uint8
        for c = 1:3
            col = rgb_pixels(:, c);
            col = (col - min(col)) / (max(col) - min(col) + 1e-8) * 255;
            rgb_pixels(:, c) = col;
        end
        
        rgb_img = uint8(reshape(rgb_pixels, [target_size, 3]));  % [224, 224, 3]
        
        % 保存
        mat_name = sprintf('class_%d_%04d.mat', class_label, s);
        save(fullfile(mat_dir, ['class_', num2str(class_label)], mat_name), ...
             'rgb_img', 'coeff');
        total_generated = total_generated + 1;
    end
end
fprintf('生成完成，共 %d 个样本。\n', total_generated);

%% ========== 2. 平衡样本 ==========
fprintf('\n========== 平衡样本 ==========\n');
rng(42);

min_count = Inf;
for class = 0:3
    mat_folder = fullfile(mat_dir, ['class_', num2str(class)]);
    d = dir(fullfile(mat_folder, '*.mat'));
    min_count = min(min_count, length(d));
end
target_num = min_count;
fprintf('每类保留 %d 个样本\n', target_num);

for class = 0:3
    mat_folder = fullfile(mat_dir, ['class_', num2str(class)]);
    mat_files = dir(fullfile(mat_folder, '*.mat'));
    current_num = length(mat_files);
    
    if current_num <= target_num
        fprintf('类别 %d: 已有 %d 个，无需删除\n', class, current_num);
        continue;
    end
    
    file_nums = zeros(current_num, 1);
    for i = 1:current_num
        num_str = regexp(mat_files(i).name, 'class_\d+_(\d+)\.mat', 'tokens', 'once');
        if ~isempty(num_str)
            file_nums(i) = str2double(num_str{1});
        end
    end
    file_nums = file_nums(file_nums > 0);
    
    keep_nums = sort(randsample(file_nums, target_num));
    delete_nums = setdiff(file_nums, keep_nums);
    
    for d = 1:length(delete_nums)
        mat_name = sprintf('class_%d_%04d.mat', class, delete_nums(d));
        delete(fullfile(mat_folder, mat_name));
    end
    fprintf('类别 %d: 保留 %d 个\n', class, target_num);
end

%% ========== 3. 加载并划分数据集 ==========
fprintf('\n========== 划分数据集 ==========\n');

all_data = {};
all_labels = [];

for class = 0:3
    mat_folder = fullfile(mat_dir, ['class_', num2str(class)]);
    mat_files = dir(fullfile(mat_folder, '*.mat'));
    
    for m = 1:length(mat_files)
        load(fullfile(mat_folder, mat_files(m).name), 'rgb_img');
        all_data{end+1} = rgb_img;  % [224, 224, 3]
        all_labels(end+1) = class;
    end
end

X = cat(4, all_data{:});  % [224, 224, 3, N]
y = all_labels(:);

% 转为 [N, 3, 224, 224] 格式 (PyTorch: NCHW)
X = permute(X, [4, 3, 1, 2]);

fprintf('总样本数: %d\n', size(X, 1));

rng(42);
idx = randperm(size(X, 1));
X = X(idx, :, :, :);
y = y(idx);

N = size(X, 1);
n_train = round(N * train_ratio);
n_val   = round(N * val_ratio);

X_train = X(1:n_train, :, :, :);
y_train = y(1:n_train);

X_val = X(n_train+1 : n_train+n_val, :, :, :);
y_val = y(n_train+1 : n_train+n_val);

X_test = X(n_train+n_val+1 : end, :, :, :);
y_test = y(n_train+n_val+1 : end);

save(fullfile(output_root, 'train_data.mat'), 'X_train', 'y_train', '-v7.3');
save(fullfile(output_root, 'val_data.mat'),   'X_val',   'y_val',   '-v7.3');
save(fullfile(output_root, 'test_data.mat'),  'X_test',  'y_test',  '-v7.3');

fprintf('\n========== 完成 ==========\n');
fprintf('输出目录: %s\n', output_root);
fprintf('训练集: %d, 验证集: %d, 测试集: %d\n', n_train, n_val, N - n_train - n_val);
fprintf('图像格式: RGB [224, 224, 3], 8通道PCA降维\n');
