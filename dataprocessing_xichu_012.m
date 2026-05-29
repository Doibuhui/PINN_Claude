% dataprocessing_xichu_012.m
% CWRU 多负载跨工况故障诊断
% 训练集: 0HP + 1HP, 测试集: 2HP (未见负载 → 考验泛化能力)
% 10类轴承故障: Normal / IR007/014/021 / OR007/014/021 / Ball007/014/021
% CWT时频图, 3通道 (幅值+cos相位+sin相位), 统一 resize 224×224

clear; clc; close all;

%% ========== 参数设置 ==========
window_len         = 512;
overlap            = 256;    % 50% overlap
step               = window_len - overlap;
sampling_rate      = 12000;
wavelet            = 'amor';
voices_per_octave  = 48;
target_size        = [224 224];
% target_num 不再写死，运行时会自动取每负载下各类的最少样本数

% 训练集内部划分比例
train_split = 0.85;             % 训练:验证 = 85:15

%% ========== 路径配置 ==========
% 三个负载的数据源
load_dirs = {
    'C:\Users\ClearNight\Desktop\10series\CWRU0HP';
    'C:\Users\ClearNight\Desktop\10series\CWRU1HP';
    'C:\Users\ClearNight\Desktop\10series\CWRU2HP';
};

output_root = 'C:\Users\ClearNight\Desktop\PINN\CWRU_CrossLoad';
temp_dir    = fullfile(output_root, 'TempMatrices');

%% ========== 标签映射 (10类, 每负载不同文件编号) ==========
%      列1:0HP     列2:1HP     列3:2HP    列4:类标签  列5:说明
label_map = {
     '97',       '98',       '99',       0,        'Normal';
    '105',      '106',      '107',       1,        'IR007';
    '169',      '170',      '171',       2,        'IR014';
    '209',      '210',      '211',       3,        'IR021';
    '130',      '131',      '132',       4,        'OR007';
    '197',      '198',      '199',       5,        'OR014';
    '234',      '235',      '236',       6,        'OR021';
    '118',      '119',      '120',       7,        'Ball007';
    '185',      '186',      '187',       8,        'Ball014';
    '222',      '223',      '224',       9,        'Ball021';
};

NUM_CLASSES = size(label_map, 1);

%% ================================================================
% 第1步: 三个负载各自生成 CWT 矩阵 (保存到临时目录)
%% ================================================================
fprintf('========== 第1步: 逐负载生成 CWT ==========\n');

% 清空临时目录
if exist(temp_dir, 'dir')
    rmdir(temp_dir, 's');
end
mkdir(temp_dir);

% 创建临时子目录: TempMatrices/load{0,1,2}/class{0-9}/
for ld = 0:2
    for c = 0:NUM_CLASSES-1
        mkdir(fullfile(temp_dir, sprintf('load%d', ld), sprintf('class%d', c)));
    end
end

rng(42);

for ld = 1:3
    data_dir = load_dirs{ld};
    load_idx = ld - 1;  % 0, 1, 2
    
    fprintf('\n--- 负载 %d: %s ---\n', load_idx, data_dir);
    
    for k = 1:size(label_map, 1)
        filename    = [label_map{k, load_idx + 1}, '.mat'];  % 对应负载列
        class_label = label_map{k, 4};                       % 类别标签
        file_path   = fullfile(data_dir, filename);
        
        if ~exist(file_path, 'file')
            warning('  文件不存在: %s', file_path);
            continue;
        end
        
        % 加载信号 (精确匹配文件ID避免多变量混淆，如99.mat同时含X098/X099)
        data_struct = load(file_path);
        vn = fieldnames(data_struct);
        file_id = filename(1:end-4);  % '99.mat' -> '99'
        expected_name = ['X' sprintf('%03d', str2double(file_id)) '_DE_time'];
        de_var = '';
        for v = 1:length(vn)
            if strcmp(vn{v}, expected_name)
                de_var = vn{v};
                break;
            end
        end
        % fallback: 文件名不一致时退回到模糊匹配
        if isempty(de_var)
            for v = 1:length(vn)
                if contains(vn{v}, 'DE_time')
                    de_var = vn{v};
                    break;
                end
            end
        end
        if isempty(de_var)
            warning('  未找到DE_time: %s', filename);
            continue;
        end
        signal  = data_struct.(de_var);
        sig_len = length(signal);
        
        num_samples = floor((sig_len - window_len) / step) + 1;

        fprintf('  [负载%d 类别%2d] %s: %d 样本\n', ...
                load_idx, class_label, filename, num_samples);
        
        for s = 1:num_samples
            start_idx = (s-1) * step + 1;
            seg = signal(start_idx : start_idx + window_len - 1);
            
            % CWT (Morlet) — 保留幅值 + 连续相位 (cos/sin, 消除±π断崖)
            [cfs, ~] = cwt(seg, wavelet, sampling_rate, ...
                           'VoicesPerOctave', voices_per_octave);
            mag  = abs(cfs);
            ph   = angle(cfs);           % 相位, [-π, π]
            pcos = cos(ph);              % [-1, 1] 连续, 无断崖
            psin = sin(ph);              % [-1, 1] 连续

            % 幅值线性归一化到 [0, 1]; 相位 cos/sin 本身在 [-1,1] 无需归一化
            mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));

            % 保存为独立 .mat 文件 (3通道: 幅值 + 相位cos + 相位sin)
            save_path = fullfile(temp_dir, sprintf('load%d', load_idx), ...
                        sprintf('class%d', class_label), ...
                        sprintf('class_%d_%04d.mat', class_label, s));
            save(save_path, 'mag_norm', 'pcos', 'psin');
        end
    end
    
    % 自动计算该负载下最少样本数，按时间顺序保留前 N 个
    min_count = Inf;
    for c = 0:NUM_CLASSES-1
        class_folder = fullfile(temp_dir, sprintf('load%d', load_idx), sprintf('class%d', c));
        mat_files = dir(fullfile(class_folder, '*.mat'));
        min_count = min(min_count, length(mat_files));
    end
    fprintf('  平衡负载 %d 至每类 %d (自动取各类最少样本数)\n', load_idx, min_count);

    for c = 0:NUM_CLASSES-1
        class_folder = fullfile(temp_dir, sprintf('load%d', load_idx), sprintf('class%d', c));
        mat_files = dir(fullfile(class_folder, '*.mat'));

        if length(mat_files) <= min_count
            continue;
        end

        % 按样本序号排序（时间顺序），保留前 min_count 个
        nums = cellfun(@(x) sscanf(x, 'class_%*d_%d.mat'), {mat_files.name});
        [~, sort_idx] = sort(nums);
        sorted_nums = nums(sort_idx);
        % 丢弃末尾样本（晚时间段），减少 train/val 时间泄露
        delete_nums = sorted_nums(min_count+1:end);
        for i = 1:length(delete_nums)
            delete(fullfile(class_folder, ...
                  sprintf('class_%d_%04d.mat', c, delete_nums(i))));
        end
    end
end

%% ================================================================
% 第2步: 时域连续划分 → 训练(0HP+1HP) / 验证(来自训练) / 测试(2HP)
% 每类样本按时间顺序前85%→训练、后15%→验证，避免50%重叠窗口泄露
%% ================================================================
fprintf('\n========== 第2步: 跨负载时域连续划分 ==========\n');

train_data_cell  = {};
train_labels_all = [];
val_data_cell    = {};
val_labels_all   = [];

for ld = 0:1  % 仅 0HP 和 1HP
    for c = 0:NUM_CLASSES-1
        class_folder = fullfile(temp_dir, sprintf('load%d', ld), sprintf('class%d', c));
        mat_files = dir(fullfile(class_folder, '*.mat'));

        % 按样本序号（生成顺序 = 时间先后）排序
        nums = cellfun(@(x) sscanf(x, 'class_%*d_%d.mat'), {mat_files.name});
        [~, sort_idx] = sort(nums);

        n_samples = length(mat_files);
        n_tr = round(n_samples * train_split);

        % 前 n_tr 个（时间较早）→ 训练集
        for m = 1:n_tr
            idx = sort_idx(m);
            load(fullfile(class_folder, mat_files(idx).name), 'mag_norm', 'pcos', 'psin');
            mag_img = imresize(mag_norm, target_size);
            cos_img = imresize(pcos, target_size);
            sin_img = imresize(psin, target_size);
            img = cat(3, mag_img, cos_img, sin_img);  % [224, 224, 3]
            train_data_cell{end+1}  = img;  %#ok<AGROW>
            train_labels_all(end+1) = c;    %#ok<AGROW>
        end

        % 后 (n_samples - n_tr) 个（时间较晚）→ 验证集
        for m = n_tr+1:n_samples
            idx = sort_idx(m);
            load(fullfile(class_folder, mat_files(idx).name), 'mag_norm', 'pcos', 'psin');
            mag_img = imresize(mag_norm, target_size);
            cos_img = imresize(pcos, target_size);
            sin_img = imresize(psin, target_size);
            img = cat(3, mag_img, cos_img, sin_img);
            val_data_cell{end+1}  = img;  %#ok<AGROW>
            val_labels_all(end+1) = c;    %#ok<AGROW>
        end
    end
end

fprintf('  训练总样本: %d\n', length(train_labels_all));
fprintf('  验证总样本: %d\n', length(val_labels_all));

% --- 加载测试数据 (负载2) ---
fprintf('加载测试数据 (2HP)...\n');
test_data_cell  = {};
test_labels_all = [];

for c = 0:NUM_CLASSES-1
    class_folder = fullfile(temp_dir, 'load2', sprintf('class%d', c));
    mat_files = dir(fullfile(class_folder, '*.mat'));

    % 按时间顺序排序后加载
    nums = cellfun(@(x) sscanf(x, 'class_%*d_%d.mat'), {mat_files.name});
    [~, sort_idx] = sort(nums);

    for m = 1:length(mat_files)
        idx = sort_idx(m);
        load(fullfile(class_folder, mat_files(idx).name), 'mag_norm', 'pcos', 'psin');
        mag_img = imresize(mag_norm, target_size);
        cos_img = imresize(pcos, target_size);
        sin_img = imresize(psin, target_size);
        img = cat(3, mag_img, cos_img, sin_img);
        test_data_cell{end+1}  = img;  %#ok<AGROW>
        test_labels_all(end+1) = c;    %#ok<AGROW>
    end
end
fprintf('  测试总样本: %d\n', length(test_labels_all));

% --- 检查各类分布 ---
fprintf('\n各类别样本分布:\n');
fprintf('  类别 | 训练 | 验证 | 测试\n');
fprintf('  -----|------|------|-----\n');
train_counts = histcounts(train_labels_all, 0:10);
val_counts   = histcounts(val_labels_all, 0:10);
test_counts  = histcounts(test_labels_all, 0:10);
for c = 0:NUM_CLASSES-1
    fprintf('  %4d | %4d | %4d | %4d\n', c, train_counts(c+1), val_counts(c+1), test_counts(c+1));
end

% --- 各自内部打乱 ---
rng(42);
shuffle_tr = randperm(length(train_labels_all));
X_train = cat(4, train_data_cell{:});
y_train = train_labels_all(:);
X_train = X_train(:, :, :, shuffle_tr);
y_train = y_train(shuffle_tr);

rng(43);
shuffle_val = randperm(length(val_labels_all));
X_val = cat(4, val_data_cell{:});
y_val = val_labels_all(:);
X_val = X_val(:, :, :, shuffle_val);
y_val = y_val(shuffle_val);

X_test = cat(4, test_data_cell{:});
y_test = test_labels_all(:);

% 转为 PyTorch NCHW 格式: [H, W, C, N] → [N, C, H, W]
X_train = permute(X_train, [4, 3, 1, 2]);
X_val   = permute(X_val,   [4, 3, 1, 2]);
X_test  = permute(X_test,  [4, 3, 1, 2]);

% --- 保存 ---
fprintf('\n保存数据集...\n');
save(fullfile(output_root, 'train_data.mat'), 'X_train', 'y_train', '-v7.3');
save(fullfile(output_root, 'val_data.mat'),   'X_val',   'y_val',   '-v7.3');
save(fullfile(output_root, 'test_data.mat'),  'X_test',  'y_test',  '-v7.3');

% --- 清理临时文件 ---
fprintf('清理临时文件...\n');
rmdir(temp_dir, 's');

%% ================================================================
fprintf('\n========== 完成 ==========\n');
fprintf('输出目录: %s\n', output_root);
fprintf('训练集:   %d (0HP + 1HP, 85%%)\n', size(X_train, 1));
fprintf('验证集:   %d (0HP + 1HP, 15%%)\n', size(X_val, 1));
fprintf('测试集:   %d (2HP, 100%%)\n', size(X_test, 1));
fprintf('图像尺寸: %d×%d, 3通道 (幅值+相位cos+相位sin)\n', target_size(1), target_size(2));
fprintf('类别数:   %d\n', NUM_CLASSES);
fprintf('\n训练配置默认: 0HP+1HP → 测试 2HP (跨负载泛化)\n');
