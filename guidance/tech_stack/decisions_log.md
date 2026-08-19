# 重大技术调整记录（Decisions Log）

## D-001 初始建立指导文件夹

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：依据 Constitution.md 与用户实验条件建立项目指导文件。
- 影响范围：全项目。
- 版本：guidance 0.1.0

## D-002 参考电极默认 A1 而不是说明书默认 Cz

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：Cz 是 8 个记录通道之一，不能同时作为 REF；说明书默认 REF=Cz 只适用于
  其他 Montage。宪法红线要求记录原始参考与通道位置。
- 影响范围：`config/channel_config.json`、采集元数据、离线重参考。
- 回滚：若硬件确实使用 Cz 作参考，必须把 Cz 从记录通道中移除并新增可用记录电极，
  并更新本记录。

## D-003 采样率选择 500 Hz

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：BS8A 支持 500Hz；高于常规 ShallowConvNet 250Hz，保留更多 ERP 时间细节；
  算力优先级低。模型输入采样率可在后端重采样，保留原始 500Hz EDF。
- 影响范围：前端默认采集参数、后端重采样参数。

## D-004 SOA 对齐文献 ISI=1500ms

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：文献描述 inter-stimulus interval 1500ms 未严格定义 onset/offset。
  前端默认刺激呈现 200ms + 空白 1300ms，使 onset-to-onset=1500ms。
- 影响范围：前端范式。
- 备注：如需严格 offset-to-onset=1500ms，使用 `--stimulus-ms 200 --blank-ms 1500`。

## D-005 结合 BrainSync SDK 特征改进（联网调研后）

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 依据：
  - BrainSync SDK `subscribe_eeg_data` 要求 batch_size 为 250 的整数倍；
  - `EegDataPacket` 提供 24-bit ADC、`to_microvolts(gain)`、`delta_time_us()`、
    `seq_num`、`status.leadoff_status`、`trig_in_status`、`is_impedance_mode()`；
  - `ChannelConfig`/`start_edf_recording` 可写电极标签与 EDF annotation。
- 变更内容：
  1. 前端真实设备 batch_size 改为 250，并对采样点做包间插值；
  2. 前端逐批记录导联脱落/阻抗模式状态，写入 events.jsonl 与 session.json；
  3. 后端将 leadoff_status != 255 或 impedance-mode 试次标记为坏试次；
  4. 预处理低通由 40Hz 收紧到 20Hz（P300 能量集中在 delta/theta，干电极肌电噪声主要在 20Hz 以上）；
  5. 增加 xDAWN 空间滤波通道（target 2 成分 + non-target 1 成分 + 原始 8 通道）；
  6. ShallowConvNet temporal kernel 从 0.10s 增至 0.20s，filters 40→60；
  7. 增加 mixup(α=0.2) 与更强的通道 dropout/噪声增强；
  8. block 预测增加 Bayesian softmax confidence 与 logit margin。
- 影响范围：frontend/acquisition.py、experiment.py；backend/io.py、preprocess.py、
  xdawn.py、dataset.py、train.py、model.py、scoring.py、config/*.json。
- 回滚：将 `config/preprocessing.json` 的 `xdawn_enable=false`、`lowpass_hz=40`；
  将 `config/train.json` 的 `mixup_alpha=0`、`temporal_filters=40`、`kernel_time_s=0.10`。
