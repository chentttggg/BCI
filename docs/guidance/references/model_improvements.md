# 模型改进调研笔记（联网检索 + BrainSync SDK 实测）

## 关键依据

1. **ShallowConvNet 原始结构**（Schirrmeister et al., 2017）：
   temporal conv(40, 1×25) → spatial conv(40, C×1) → BN → square → avg-pool(75/15)
   → log → dropout → conv classifier。本实现保持该主干。
   - 参考实现：https://raw.githubusercontent.com/braindecode/braindecode/refs/heads/master/braindecode/models/shallow_fbcsp.py
2. **xDAWN + 深度网络**：xDAWN 增强 ERP 信噪比后再接 CNN 的 cascade 结构
   在 VEP/P300 检测中有效（Cascade xDAWN EEGNet, IEEE 2024）。
3. **小样本 ERP 增广**：时间抖动、幅度缩放、通道 dropout、噪声注入以及
   mixup 是 EEG 深度模型常用正则化手段；本实验单被试 target 试次少，必须强正则。
4. **动态停止/Bayesian 聚合**：P300 拼写器常用逐试次证据累积和置信度阈值，
   本项目的 mean-logit 分数即等先验下的对数后验比，softmax 后可作为 block 置信度。

## BrainSync SDK 数据特征（本机 `brainsync_sdk==0.3.0` 实测）

| 特征 | 含义 | 本项目利用 |
| --- | --- | --- |
| `EegDataPacket.adc_values` | 8 通道 24-bit 有符号 ADC | 前端以 `to_microvolts(Gain24)` 保存 µV |
| `delta_time_us()` | 采样时间差（µs） | 事件采样点插值、丢包 QC |
| `seq_num` | 数据包序号 | 丢包检测与事件索引 |
| `status.leadoff_status` | 导联脱落状态位 | 后端将非 255 试次标记为坏试次 |
| `status.trig_in_status` | 触发输入状态 | 记录，支持 TriggerHub 验证 |
| `is_impedance_mode()` | 是否阻抗检测模式 | 前端跳过非 EEG 数据包 |
| `subscribe_eeg_data(batch_size)` | 必须为 250 的整数倍 | 前端 batch_size 设为 250 |
| `ChannelConfig` | 电极标签/REF/GND 元数据 | LSL 通道元数据与 EDF 标签一致 |
