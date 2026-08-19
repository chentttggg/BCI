# 实验范式与操作协议 — 猜数字 P300

## 1. 文献要点

- 文献：R. Moucek et al., *Event-related potential data from a guess the number
  brain-computer interface experiment on school children*, Scientific Data 4:160121 (2017).
- 任务：受试者从 1–9 中默想一个数字（target）；屏幕随机呈现 1–9；实验者观察 ERP 猜测。
- 刺激：黑底白色数字；inter-stimulus interval 1500 ms。
- 原始文献记录通道：Fz, Cz, Pz（+参考/接地/EOG）；epoch 窗口 -500 ~ +1000 ms；
  基线 -500 ~ 0 ms；漂移趋势建议 0.5 Hz 高通处理。
- 本实验扩展：8 通道 Fz, Cz, P3, Pz, P4, PO7, PO8, Oz；ShallowConvNet 自动解码。

## 2. 本实验参数（默认）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 采样率 | 500 Hz | 原始 EDF 保存；后端可降采样 |
| 增益 | Gain24 | 与 SDK 示例一致；以 session.json 为准 |
| 刺激呈现 | 200 ms | 黑底白字，字体约 15% 屏高 |
| 空白间隔 | 1300 ms | 使 onset-to-onset = 1500 ms |
| 单 block 试次 | 45 = 9 × 5 | 每个数字每 block 出现 5 次 |
| 默认 block 数 | 6 | 约 30 个 target 试次 |
| 注视点 | 500 ms（block 开始前） | 屏幕中央 “+” |
| epoch | -200 ~ +1000 ms | 与文献一致 |
| 目标试次定义 | 刺激数字 == 受试者默想数字 | 由 session.json `target_number` 决定 |

## 3. 前端操作流程

1. 连接设备，确认固件版本；检查阻抗 < 10 kΩ（建议 < 5 kΩ）。
2. 确认 `config/channel_config.json` 的通道顺序、REF/GND 与物理连线一致。
3. 运行前端：
   ```bash
   # mock 自测
   python -m frontend.main --mock --target 7 --output-dir data/raw
   # 真实设备
   python -m frontend.main --device --subject P01 --session 001 --target 7 --output-dir data/raw
   ```
4. 告知受试者规则：默想数字、默数目标出现次数、少眨眼、不动。
5. 每个 block 结束后查看预测与 ERP 图；建议至少采集 4–6 个 block。
6. 结束后按 `Esc` 或关闭窗口；确认 EDF、events.jsonl、session.json 已生成并只读归档。

## 4. 事件编码

| marker | 含义 |
| --- | --- |
| `session_start` / `session_stop` | 采集会话边界 |
| `target/{n}` | 本 session 受试者默想数字（记录用） |
| `block_start/{i}` / `block_end/{i}` | block 边界 |
| `stim_on/{n}` | 数字 n 刺激开始（关键 onset） |
| `stim_off/{n}` | 数字 n 刺激结束 |
| `artifact/{note}` | 实验者手动标记（保留，不删除） |

每个 marker 同时写入：
- LSL 输出流 `GuessNumberMarkers`（字符串 marker）；
- EDF+ annotation（若 EDF 录制开启）；
- `events.jsonl`（含 monotonic 时间、LSL 时间、预期 onset、block、trial、数字）。

## 5. 受试者与操作者记录

`session.json` 至少包含：participant_id、session_id、日期、target_number、
采样率、增益、通道映射、ref_label、gnd_label、软件版本、git commit、随机种子。
