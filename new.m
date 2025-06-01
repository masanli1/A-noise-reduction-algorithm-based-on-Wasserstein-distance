clear all; close all; clc;

%% =1. 信号预处理模块 ====================
% 参数配置
target_fs = 24000;                % 目标采样率(Hz)
noisy_file = 'test_audio.wav'; % 含噪音频文件
clean_ref_file = 'resampled.wav';  % 纯净参考文件(可选)

% 音频读取与重采样
[noisy_signal, fs] = audioread(noisy_file);
fprintf('\n=== 输入信号特征 ===\n');
fprintf('原始采样率: %d Hz\n文件时长: %.2f秒\n', fs, length(noisy_signal)/fs);

% 重采样处理
if fs ~= target_fs
    fprintf('[预处理] 重采样中：%dHz → %dHz\n', fs, target_fs);
    noisy_signal = resample(noisy_signal, target_fs, fs);
    fs = target_fs;
end

% 单声道转换与标准化
x = mean(noisy_signal, 2);
x = x(:);  % 强制列向量
x = x/max(abs(x))*0.95;  % 初始标准化

% 读取参考信号(用于评估)
clean_signal = [];
if exist(clean_ref_file, 'file')
    [temp_clean, fs_clean] = audioread(clean_ref_file);
    if fs_clean ~= target_fs
        temp_clean = resample(temp_clean, target_fs, fs_clean);
    end
    clean_signal = mean(temp_clean, 2);
    clean_signal = clean_signal(:);
end

%% =2. 信号预调整模块 ====================
% 小波分解参数
wname = 'sym8';                     % 小波基类型
max_level = floor(log2(length(x)) - 3);  % 最大分解层数
level = min(6, max_level);          % 实际使用层数
required_length = 2^level * ceil(length(x)/2^level);  % 合法长度

% 信号长度调整(补零/截断)
if required_length > length(x)
    x(end+1:required_length) = 0;  % 尾部补零
    fprintf('[预处理] 信号补零: %d → %d点\n', length(x), required_length);
else
    x = x(1:required_length);      % 头部截断
    fprintf('[预处理] 信号截断: %d → %d点\n', length(x), required_length);
end

% 参考信号同步调整
if ~isempty(clean_signal)
    clean_signal = clean_signal(1:required_length);
    clean_signal(end+1:required_length) = 0;
end

%% = 3. 人声特征探测系统 ====================
fprintf('\n=== 人声特征分析 ===\n');
[swa, swd] = swt(x, level, wname);  % 平稳小波分解

% 参数配置
voice_params = struct(...
    'grad_window', 5,   ...       % 梯度计算窗口
    'stable_thresh', 0.12,...     % 稳定性阈值
    'dist_thresh', 0.3,  ...      % Wasserstein距离阈值
    'morph_radius', 3);        % 形态学处理半径

% 初始化特征矩阵
voice_map = zeros(size(swd));    % 人声概率分布
grad_buffer = cell(1, level);
for k = 1:level
    grad_buffer{k} = [];  % 明确初始化
end

for k = 1:level
    fprintf('正在分析第%d层...\n', k);
    coeffs = swd(k,:);
    noise_ref = swd(1,:);       % 噪声参考层
    
    % 分块处理(256样本/块)
    block_size = 256;
    for i = 1:block_size:length(coeffs)
        idx = i:min(i+block_size-1, length(coeffs));
        block_data = coeffs(idx);
        
        % 增强Wasserstein计算(含时间梯度)
        [current_dist, grad] = enhanced_wasserstein(...
            block_data, noise_ref, grad_buffer{k});
        
        % ==== 修正部分开始 ====
        current_grad = grad_buffer{k};
        max_history = voice_params.grad_window * 2; 
        if length(current_grad) >= voice_params.grad_window
            current_grad = current_grad(end-voice_params.grad_window+2:end);
        end
        grad_buffer{k} = [current_grad, grad];
        % ==== 修正部分结束 ====
        
        % 人声特征判定
        stability = 1 - mean(abs(grad_buffer{k}))/0.3;
        is_voice = (current_dist > voice_params.dist_thresh) && ...
                   (stability > voice_params.stable_thresh);
        
        voice_map(k,idx) = is_voice * stability;
    end
        
    % 形态学后处理
    voice_map(k,:) = bwareaopen(voice_map(k,:) > 0, 3);  % 去除孤立点
    se = strel('rectangle', [1, voice_params.morph_radius]);
    voice_map(k,:) = imclose(voice_map(k,:), se);        % 闭合操作
end

%% = 4. 动态加权降噪处理 ====================
fprintf('\n=== 开始动态降噪 ===\n');
retain_ratio = 0.8;  % 保留系数比例

for k = 1:level
    coeffs = swd(k,:);
    
    % 权重计算三要素
    base_weight = compute_base_weight(coeffs, swd(1,:));  % 基础降噪权重
    voice_boost = 1 + 2*voice_map(k,:);                  % 人声增强因子
    noise_mask = detect_burst_noise(grad_buffer{k});     % 突发噪声掩膜
    
    % 复合权重矩阵
    final_weight = base_weight .* voice_boost;
    final_weight(noise_mask) = final_weight(noise_mask) * 0.2;  % 突发噪声抑制
    
    % 有损系数选择
    [~, sort_idx] = sort(abs(coeffs), 'descend');
    retain_idx = sort_idx(1:round(retain_ratio*length(coeffs)));
    retain_mask = zeros(size(coeffs));
    retain_mask(retain_idx) = 1;
    
    % 应用处理
    swd(k,:) = coeffs .* final_weight .* retain_mask;
end

%% = 5. 信号重构与后处理 ====================
% 小波重构
xd = iswt(swa, swd, wname);

% 能量标准化
xd = xd * (rms(x)/rms(xd));  % RMS匹配
xd = xd/max(abs(xd))*0.95;    % 峰值限制

% 去除补零部分
if exist('pad_length', 'var')
    xd = xd(1:end-pad_length);
    x = x(1:end-pad_length);
end

% 保存结果
audiowrite('enhanced_audio.wav', xd, fs);
fprintf('\n[系统] 降噪文件已保存: enhanced_audio.wav\n');

%% = 6. 可视化分析系统 ====================
figure('Name','智能降噪分析仪','Position',[100 100 1400 900],'Color','w')

% 时域波形对比
subplot(3,2,[1,2])
plot((1:length(x))/fs, x, 'Color',[0.5 0.5 0.5], 'LineWidth',0.8)
hold on
plot((1:length(xd))/fs, xd, 'b', 'LineWidth',1.2)
title('时域波形对比'), xlabel('时间 (s)'), ylabel('幅值')
legend('原始含噪','降噪输出','Location','best')
grid on, xlim([0 length(x)/fs])

% 人声特征可视化
subplot(3,2,3)
imagesc(voice_map)
title('人声特征分布'), xlabel('时间帧'), ylabel('分解层数')
colorbar

% 功率谱对比
subplot(3,2,4)
[P_orig, F] = pwelch(x, hann(1024), 512, 1024, fs);
[P_den, ~] = pwelch(xd, hann(1024), 512, 1024, fs);
semilogy(F, P_orig, 'r', F, P_den, 'b')
title('功率谱密度对比'), xlabel('频率 (Hz)'), legend('原始','降噪')
grid on, xlim([0 8000])

% 语谱图对比
subplot(3,2,5)
spectrogram(x, 512, 480, 512, fs, 'yaxis')
title('原始信号语谱图'), clim([-80 20])

subplot(3,2,6)
spectrogram(xd, 512, 480, 512, fs, 'yaxis')
title('降噪信号语谱图'), clim([-80 20])

%% = 7. 质量评估模块 ====================
if ~isempty(clean_signal)
    min_len = min(length(clean_signal), length(xd));
    clean = clean_signal(1:min_len);
    noisy = x(1:min_len);
    denoised = xd(1:min_len);
    
    % 计算客观指标
    SNR_improve = snr(denoised, clean) - snr(noisy, clean);
    [pesq_score, ~] = pesq(clean, denoised, fs);
    
    fprintf('\n===== 质量评估报告 =====\n');
    fprintf('SNR提升量: \t%.2f dB\n', SNR_improve);
    fprintf('PESQ评分: \t%.2f/4.5\n', pesq_score);
    
    % MFCC特征分析
    mfcc_params = struct('fs',fs, 'num_ceps',13);
    mfcc_clean = mfcc(clean, mfcc_params);
    mfcc_den = mfcc(denoised, mfcc_params);
    mfcc_dist = mean(sqrt(sum((mfcc_clean - mfcc_den).^2, 2)));
    fprintf('MFCC失真度: \t%.3f\n', mfcc_dist);
else
    fprintf('\n[提示] 未检测到参考信号，跳过客观评估\n');
end

%% = 本地函数定义 ====================
function [dist, grad] = enhanced_wasserstein(data, ref, hist_grad)
% 增强型Wasserstein距离计算(含时间梯度)
% 输入：
%   data - 当前数据块
%   ref - 参考分布数据
%   hist_grad - 历史梯度数据
% 输出：
%   dist - Wasserstein距离
%   grad - 梯度变化量

% 分位数计算(避免端点噪声)
q_points = linspace(0.1, 0.9, 50);
qc = quantile(data, q_points);
qr = quantile(ref, q_points);

% L1-Wasserstein距离
dist = mean(abs(qc - qr));

% 梯度计算(相对于历史数据)
if nargin > 2 && ~isempty(hist_grad)
    % 安全获取最近3点（动态适应历史长度）
    avail_len = length(hist_grad);
    start_idx = max(1, avail_len - 2);  % 关键修正点
    prev_dist = mean(hist_grad(start_idx:end));
else
    prev_dist = 0;  % 默认值
end

grad = dist - prev_dist;
end

function weights = compute_base_weight(coeffs, noise_ref)
% 基础降噪权重计算
% 输入：
%   coeffs - 当前层系数
%   noise_ref - 噪声参考层
% 输出：
%   weights - [0-1]权重矩阵

block_size = 256;
weights = ones(size(coeffs));

for i = 1:block_size:length(coeffs)
    idx = i:min(i+block_size-1, length(coeffs));
    block = coeffs(idx);
    
    % 能量指标
    energy = mean(abs(block));
    
    % Wasserstein指标
    [dist, ~] = enhanced_wasserstein(block, noise_ref);
    
    % 权重计算公式
    w = 1 - exp(-2.5*(dist^1.2)) / (1 + 0.8*energy);
    weights(idx) = max(0, min(w, 1));  % 限制在[0,1]范围
end
end

function mask = detect_burst_noise(grad_hist)
% 突发噪声检测
% 输入：
%   grad_hist - 梯度历史数据
% 输出：
%   mask - 噪声区域掩膜

window_size = 5;  % 移动检测窗口
threshold = 0.25; % 梯度突变阈值

% 计算移动极值
grad_abs = abs(grad_hist);
mov_max = movmax(grad_abs, [window_size-1 0]);

% 生成噪声掩膜
mask = (mov_max > threshold);
end