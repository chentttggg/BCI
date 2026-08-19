# 质量门禁（QC Gates）

> 所有阈值都是“项目默认”。修改阈值必须写入 `decisions_log.md`。
> 未达标数据**不得**进入下一阶段，必须标记、报告或人工审核。
> 代码执行位置：QG-0 由实验前人工核对；QG-1/QG-2 由 `backend.ingest` 执行；
> QG-3 由 `backend.preprocess.prepare_session` 执行；QG-4 由 `backend.train`
> 生成报告，未达到 `mission/project_brief.md` 成功标准时不得发布模型。

## QG-0 采集前

| 检查项 | 阈值 |
| --- | --- |
| 电极阻抗 | 每个 EEG 通道 < 10 kΩ；建议 < 5 kΩ |
| 通道映射 | `channel_config.json` 与物理连线完全一致 |
| 采样率/增益 | 与 session.json 一致（默认 250Hz / Gain24） |
| 参考/接地 | ref_label、gnd_label 已填写且不等于任何 EEG 通道 |
| 存储空间 | 预期 EDF 大小可容纳 |

## QG-1 采集完整性

| 检查项 | 阈值 |
| --- | --- |
| EDF 可读 | 无读文件错误，`n_samples > 0` |
| 通道 | 8 个 EEG 通道，标签集合 = Fz, Cz, P3, Pz, P4, PO7, PO8, Oz |
| 采样率 | 与 `preprocessing.json` 的 `raw_sfreq` 一致（默认 250 Hz，容差 0.5 Hz） |
| 刺激事件 | `stim_on` 数量 = blocks × 9 × repetitions；每个 block 内 1–9 各出现 repetitions 次 |
| 目标数字 | `session.json` 中存在 `target_number`（1–9） |
| 元数据 | session.json 含 ref_label、gnd_label、channel list、gain、seed、软件版本 |

## QG-2 原始数据初筛

| 检查项 | 阈值 |
| --- | --- |
| 平坦/导联脱落通道 | 每通道 `flat_frac ≤ 20%` |
| 满量程/轨饱和通道 | 每通道 `rail_frac ≤ 20%` |
| 完全重复通道 | 无 exact-duplicate channel group |
| marker 对齐 | events.jsonl vs EDF annotation 的 stim_on 时间差 `median_abs_ms ≤ 20 ms` |

说明：QG-2 只标记问题，不删除、不插值。`ingest` 生成的 `issues` 非空时
不得进入训练。

## QG-3 预处理与伪迹门禁

`prepare_session()` 完成后 `sidecar.qc_pass == true` 才允许训练/预测：

| 检查项 | 阈值 |
| --- | --- |
| 连续数据坏导 | ≤ 1 个（`max_bad_channels=1`） |
| 试次峰峰值 | 好导中 `ptp ≤ 150 µV` |
| 试次绝对幅值 | 好导中 `abs ≤ 120 µV` |
| 坏试次比例 | ≤ 30%（`max_bad_epoch_ratio=0.30`） |
| BrainSync lead-off/impedance | `leadoff_status==255` 且非 impedance mode；否则试次标记为坏 |
| 事件裁剪 | 所有 epoch 均落在连续数据内；裁剪失败的事件写入 `dropped_event_indices` |

坏导处理记录为 “CAR 后置零 + QC 列表”，坏试次保留 mask 并仅从监督训练中排除，
推理时保留并排除出聚合；任何处理不得静默删除。

## QG-4 模型发布

`train --cv` 输出以下报告，达到门槛才可发布（目标值用于调优）：

| 指标 | 门槛 | 目标 |
| --- | --- | --- |
| 9 选 1 block 准确率 | > 1/9 随机水平 | ≥ 60% |
| 目标试次 AUC | ≥ 0.70 | ≥ 0.85 |
| 输入数据哈希 | 100% 写入 `input_files.json` | 100% |
| QC 可复现 | 每个 session 有 QC 报告/侧车 | 每次必做 |
