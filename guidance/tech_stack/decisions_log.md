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
