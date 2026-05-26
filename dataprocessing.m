% cwt_matrix_only_split.m
% 功能：
%   1. 加载 CWRU .mat 文件 (12k DE, 0HP)
%   2. Morlet CWT 生成幅值矩阵（归一化到[0,1]）
%      不做伪彩色，不保存图片，只保存数值矩阵
%   3. 平衡样本至每类 118 张
%   4. 读取所有 .mat 文件，按 70/15/15 划分训练/验证/测试集
%      并保存为 train_data.mat, val_data.mat, test_data.mat
% 日期：2026-05-11

clear; clc; close all;

%% ========== 参数设置 ==========
window_len = 1024;              % 窗口长度
overlap = 0;
step = window_len - overlap;
sampling_rate = 12000;
wavelet = 'amor';
voices_per_octave = 48;
target_num = 118;               % 每类样本数

% 划分比例
train_ratio = 0.7;
val_ratio = 0.15;
test_ratio = 0.15;

%% ========== 路径 ==========
data_dir = 'C:\Users\ClearNight\Desktop\10series\CWRU0HP';
output_root = 'C:\Users\ClearNight\Desktop\PINN\CWT_Matrices';   % 只存矩阵
mat_dir = fullfile(output_root, 'ClassMatrices');

%% ========== 标签映射 ==========
label_map = {
    '97',  0;   % Normal
    '105', 1;   % IR007
    '169', 2;   % IR014
    '209', 3;   % IR021
    '130', 4;   % OR007
    '197', 5;   % OR014
    '234', 6;   % OR021
    '118', 7;   % Ball007
    '185', 8;   % Ball014
    '222', 9;   % Ball021
};

%% ========== 1. 生成 CWT 幅值矩阵并保存 .mat ==========
fprintf('========== 开始生成 CWT 幅值矩阵 ==========\n');

% 检查是否存在已有矩阵文件
need_generate = true;
if exist(fullfile(mat_dir, 'class_0'), 'dir')
    d = dir(fullfile(mat_dir, 'class_0', '*.mat'));
    if length(d) >= target_num
        need_generate = false;
        fprintf('检测到已有矩阵文件，跳过生成步骤。\n');
    end
end

if need_generate
    % 创建分类文件夹
    for i = 0:9
        mkdir(fullfile(mat_dir, ['class_', num2str(i)]));
    end

    total_generated = 0;

    for k = 1:size(label_map, 1)
        filename = [label_map{k, 1}, '.mat'];
        class_label = label_map{k, 2};
        file_path = fullfile(data_dir, filename);

        if ~exist(file_path, 'file')
            warning('文件不存在: %s，跳过', file_path);
            continue;
        end

        % 加载 DE_time 信号 (精确匹配文件ID，避免多变量混淆)
        data_struct = load(file_path);
        var_names = fieldnames(data_struct);
        file_id = filename(1:end-4);  % '97.mat' -> '97'
        expected_name = ['X' sprintf('%03d', str2double(file_id)) '_DE_time'];
        de_var = '';
        for v = 1:length(var_names)
            if strcmp(var_names{v}, expected_name)
                de_var = var_names{v};
                break;
            end
        end
        % fallback: 文件名不一致时退回到模糊匹配
        if isempty(de_var)
            for v = 1:length(var_names)
                if contains(var_names{v}, 'DE_time')
                    de_var = var_names{v};
                    break;
                end
            end
        end
        if isempty(de_var)
            warning('未找到DE_time: %s', filename);
            continue;
        end
        signal = data_struct.(de_var);
        sig_len = length(signal);

        num_samples = floor((sig_len - window_len) / step) + 1;
        if class_label == 0
            num_samples = min(num_samples, target_num);
        end

        fprintf('[类别 %d] %s: 生成 %d 个样本\n', class_label, filename, num_samples);

        for s = 1:num_samples
            start_idx = (s-1) * step + 1;
            seg = signal(start_idx : start_idx + window_len - 1);

            % CWT
            [cfs, freq] = cwt(seg, wavelet, sampling_rate, ...
                              'VoicesPerOctave', voices_per_octave);
            mag = abs(cfs);
            % 归一化到 [0,1]
            mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));

            % 保存矩阵（不再 resize，直接保留原始尺寸，Python 中会统一处理）
            mat_name = sprintf('class_%d_%04d.mat', class_label, s);
            save(fullfile(mat_dir, ['class_', num2str(class_label)], mat_name), ...
                 'mag_norm', 'freq');
            total_generated = total_generated + 1;
        end
    end
    fprintf('原始生成完成，共 %d 个样本。\n', total_generated);

    %% ========== 2. 随机平衡 ==========
    fprintf('\n========== 平衡样本至每类 %d 张 ==========\n', target_num);
    rng(42);

    for class = 0:9
        mat_folder = fullfile(mat_dir, ['class_', num2str(class)]);
        mat_files = dir(fullfile(mat_folder, '*.mat'));
        current_num = length(mat_files);

        if current_num <= target_num
            fprintf('类别 %d: 已有 %d 个，无需删除\n', class, current_num);
            continue;
        end

        % 提取序号
        file_nums = zeros(current_num, 1);
        for i = 1:current_num
            num_str = regexp(mat_files(i).name, 'class_\d+_(\d+)\.mat', 'tokens', 'once');
            if ~isempty(num_str)
                file_nums(i) = str2double(num_str{1});
            else
                file_nums(i) = NaN;
            end
        end
        file_nums = file_nums(~isnan(file_nums));

        keep_nums = sort(randsample(file_nums, target_num));
        delete_nums = setdiff(file_nums, keep_nums);

        fprintf('类别 %d: 从 %d 删除 %d 个，保留 %d 个\n', ...
                class, current_num, length(delete_nums), target_num);

        for d = 1:length(delete_nums)
            mat_name = sprintf('class_%d_%04d.mat', class, delete_nums(d));
            delete(fullfile(mat_folder, mat_name));
        end
    end
    fprintf('平衡完成。\n');
else
    fprintf('使用已有矩阵文件，跳过生成与平衡步骤。\n');
end

%% ========== 3. 加载所有样本并划分数据集 ==========
fprintf('\n========== 加载所有样本并划分数据集 ==========\n');

% 读取所有 .mat 文件，收集图像矩阵和标签
all_data = {};   % 用 cell 暂存，最后合并
all_labels = [];

for class = 0:9
    mat_folder = fullfile(mat_dir, ['class_', num2str(class)]);
    mat_files = dir(fullfile(mat_folder, '*.mat'));

    for m = 1:length(mat_files)
        load(fullfile(mat_folder, mat_files(m).name), 'mag_norm');
        % 统一 resize 到 224×224（这里用 imresize，也可保存原始尺寸，由用户决定）
        img = imresize(mag_norm, [224 224]);
        all_data{end+1} = img;          %#ok<AGROW>
        all_labels(end+1) = class;      %#ok<AGROW>
    end
end

% 转为 4D 数组 [224 224 1 N]
X = cat(4, all_data{:});   % 此时 X 为 224×224×1×N
y = all_labels(:);

fprintf('总样本数: %d\n', size(X, 4));

% 随机打乱
rng(42);
idx = randperm(size(X, 4));
X = X(:, :, :, idx);
y = y(idx);

% 计算各类划分数量
N = size(X, 4);
n_train = round(N * train_ratio);
n_val   = round(N * val_ratio);

X_train = X(:, :, :, 1:n_train);
y_train = y(1:n_train);

X_val = X(:, :, :, n_train+1 : n_train+n_val);
y_val = y(n_train+1 : n_train+n_val);

X_test = X(:, :, :, n_train+n_val+1 : end);
y_test = y(n_train+n_val+1 : end);

% 保存划分结果
save(fullfile(output_root, 'train_data.mat'), 'X_train', 'y_train', '-v7.3');
save(fullfile(output_root, 'val_data.mat'),   'X_val',   'y_val',   '-v7.3');
save(fullfile(output_root, 'test_data.mat'),  'X_test',  'y_test',  '-v7.3');

fprintf('数据集划分完成，已保存至:\n');
fprintf('  %s\n', fullfile(output_root, 'train_data.mat'));
fprintf('  %s\n', fullfile(output_root, 'val_data.mat'));
fprintf('  %s\n', fullfile(output_root, 'test_data.mat'));
fprintf('训练集: %d, 验证集: %d, 测试集: %d\n', n_train, n_val, N - n_train - n_val);