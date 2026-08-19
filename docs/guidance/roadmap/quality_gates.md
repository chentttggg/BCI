# 质量门禁（QC Gates）

> 所有阈值都是“项目默认”。修改阈值必须写入 `decisions_log.md`。
> 未达标数据**不得**进入下一阶段，必须标记、报告或人工审核。

## QG-0 采集前

| 检查项 | 阈值 |
| --- | --- |
| 电极阻抗 | 每个 EEG 通道 < 10 kΩ；建议 < 5 kΩ |
| 通道映射 | `channel_config.json` 与物理连线完全一致 |
| 采样率/增益 | 与 session.json 一致（默认 500Hz / Gain24） |
| 参考/接地 | ref_label、gnd_label 已填写且不等于任何 EEG 通道 |
| 存储空间 | 预期 EDF 大小可容纳 |

## QG-1 采集完整性
