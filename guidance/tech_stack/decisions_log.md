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

## D-006 BrainSync ChannelConfig 显式绑定项目 Montage + 降低通道 dropout

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：
  1. BrainSync SDK `ChannelConfig.default_8ch` 是 C6/C4/FC4/...，与项目无关；
     本项目必须在代码中显式构造 `ChannelConfig(labels=Fz,Cz,P3,Pz,P4,PO7,PO8,Oz,
     ref_label=A1, gnd_label=Fpz)`，避免未来误用默认 Montage。
  2. 项目只有 8 个原始电极（xDAWN 增强后 11 通道），0.15 的独立通道 dropout
     会较频繁地把 1–2 个真实电极整体置零，空间信息损失过大；而导联脱落已经由
     `leadoff_status` QC 处理。因此将通道 dropout 降至 0.05，并限制每次最多
     置零 1 个通道，保留 7/8 以上的空间覆盖。
- 影响范围：
  - 新增 `frontend/channel_config.py`、`config/brainsync_gui_channel_config.json`；
  - `frontend/main.py` 构建并传入 SDK ChannelConfig；
  - `frontend/acquisition.py` 记录 sdk_channel_config；
  - `backend/dataset.py` 增加 `channel_dropout_max_channels=1`；
  - `config/train.json` `channel_dropout_prob=0.05`。
- 回滚：`channel_dropout_prob=0.15` 且删除 max-channel 限制即可。

## D-007 按实际硬件修正 REF/GND：A1 耳部接地/REF 共用

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：用户确认本套电极帽不使用说明书默认 Fpz/AFz 接地；
  A1 是接在耳朵上的接地/REF 共用电极。
- 变更内容：
  - `config/channel_config.json`：`ref_label=A1`、`gnd_label=A1`、
    `ref_gnd_combined=true`；
  - EEG 通道仍按工作站 Channel 0..7 顺序绑定
    Fz, Cz, P3, Pz, P4, PO7, PO8, Oz；
  - `config/brainsync_gui_channel_config.json` 同步更新；
  - 前端默认 `gnd_label` 改为 A1。
- 影响范围：channel_config、前端 session 元数据、指导文档。
- 回滚：若后续硬件增加独立 GND，修改 `config/channel_config.json` 的
  `gnd_label` 与 `ref_gnd_combined=false`。

## D-008 修正 REF/GND：A1=REF，A2=GND

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：用户澄清硬件实际为 A1 接耳部 REF，A2 接耳部 GND，
  并非 A1 共用 REF/GND。
- 变更内容：
  - `config/channel_config.json`：`ref_label=A1`、`gnd_label=A2`、
    `ref_gnd_combined=false`；
  - 工作站 `multimodal_hub/channel_config.json` 与 GUI 兼容文件同步；
  - `multimodal_hub/core/channel_config.py` 默认值改为 A1/A2；
  - Montage 对话框恢复 REF≠GND 校验；
  - 前端默认 `gnd_label` 改为 A2。
- 影响范围：全部 ChannelConfig 相关文件。

## D-009 实机诊断：仅 CH1 有信号，其余 7 通道为 -8388608 满量程负轨

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 实机测试结果（COM3，fw v1.0.7.63）：
  1. `EegSignalType.TestSignal` 时 8 个 ADC 通道全部输出正常测试波，
     说明硬件 8 通道和 SDK 链路正常；
  2. `EegSignalType.Normal` 且关闭 lead-off 后，raw `adc_values` 中
     CH0/CH2–CH7 = -8388608（24-bit 负满量程），仅 CH1 有正常数据；
  3. 阻抗检测显示多数通道为 1000 kΩ（开路/超量程），读数随触碰跳动。
- 结论：不是 Gain、不是 µV 单位、不是 ChannelConfig 标签问题；
  是电极线束/HDMI 连接器/电极-头皮接触问题，当前只有 1 个通道真正接通。
- 处置：
  - 工作站代码已强制 `Normal`、逐通道 Gain24、`disable_all_eeg_leadoff_channels`；
  - 增加 `scripts/electrode_tap_test.py` 逐通道敲击测试；
  - 实验前必须重新插紧 HDMI 电极线束，并确认 8 通道阻抗均 <10 kΩ。

## D-010 数字刺激事件自动对齐到 EEG 采样点

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：需要把每个数字的 stim_on/off 信息可靠写入 EDF，并把
  LSL marker 时间戳自动对齐到 EEG 采样时钟。
- 变更内容：
  1. `EEGOutlet.push_chunk` 返回第一个样本的 LSL 时间戳；
  2. `ExperimentController` 维护最近 12 个 (recording_sample, lsl_sec) 锚点，
     marker 到来时做一阶线性拟合，将 LSL marker 时间转成 recording_sample；
  3. events.jsonl 新增 `digit`、`alignment_source`、`edf_annotation_onset_sec`；
  4. `RawEDFRecorder` 修复 EDF 多 annotation 写入：按 annotation onset
     分段写 samples，避免 pyedflib 只保留前 2 个 annotation；
  5. `backend.ingest` 输出 `marker_alignment`（事件日志 vs EDF annotation 误差）。
- 影响范围：frontend/lsl_bridge.py、experiment.py、recorder.py；
  backend/io.py、ingest.py；scripts/make_synthetic_dataset.py。
- 回滚：见 Git 历史提交。

## D-011 数字播放与采集同步、按时间分文件夹、受试者猜测记录

- 日期：2026-08-19
- 变更者：EEG 分析工程师（agent）
- 变更理由：需要“开始播放数字=开始采集”，结束后自动保存总文件与时间序列分片，
  并记录受试者口头报告的数字。
- 变更内容：
  1. 采集延迟到第一个 `stim_on` 事件才启动；
  2. 输出根目录改为 `Data/`，每次运行按 `YYYYmmdd_HHMMSS` 建独立文件夹；
  3. 结束自动保存总 EDF、events.jsonl、session.json、
     experiment_summary.json、split_manifest.json；
  4. 按 block 切分 EDF（`eeg_block_000.edf`...）；
  5. GUI 结束弹窗记录受试者猜测；headless 支持 `--subject-guess`。
- 影响范围：frontend/main.py、experiment.py、recorder.py、acquisition.py。
- 回滚：见 Git 历史提交。
