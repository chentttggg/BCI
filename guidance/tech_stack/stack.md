# Tech Stack — 猜数字 P300 项目

## 1. 语言与环境

- Python 3.10+（推荐 3.10/3.11；若使用本机 3.14 需确认依赖 wheel 可用）。
- 虚拟环境隔离：`.venv`；依赖锁定 `requirements.lock`（若生成）。
- Git 管理代码、配置、文档；原始数据不入库，仅记录清单与哈希。

## 2. 核心库

| 库 | 用途 | 备注 |
| --- | --- | --- |
| `brainsync_sdk` | BrainSync BS8A 设备连接与 EDF 录制 | 仅真实设备模式需要；缺失时前端可运行 `--mock` |
| `PySide6` + `pyqtgraph` | 实验前端 GUI 与实时波形 | 无显示环境可用 `QT_QPA_PLATFORM=offscreen` 自测 |
| `pylsl` | EEG 流、Markers 流、时间同步 | 刺激 onset 以 LSL marker + 事件日志双记录 |
| `pyedflib` | mock 模式 EDF+ 写入、后端 EDF 兜底读取 | 连续 EEG 禁止用 CSV |
| `mne` | 首选 EEG 处理/数据容器 | 按 Constitution 默认库 |
| `numpy`/`scipy`/`pandas` | 数值处理、报告表 |
| `scikit-learn` | 交叉验证划分、校准、指标 |
| `torch` | ShallowConvNet 训练与推理 | CPU 优先，算力问题优先级低 |
| `matplotlib` | QC 图、ERP 图 |

## 3. 数据与目录标准（BIDS 简化版）

```text
data/
  raw/                 # 只读原始区，只放采集程序写入的 EDF、event log
  derived/
    sessions/          # 标准化 FIF/Epoched、清洗副本
    features/          # 模型输入数组
    models/            # 模型权重、预处理参数、provenance
    reports/           # QC HTML/PNG/JSON
    predictions/       # 推理结果
  manifest.jsonl       # 每个原始文件的 SHA-256、大小、采集参数、操作者
```

- 命名约定：
  `sub-{participant}_ses-{session}_task-guessnumber_run-{run}_eeg.edf`
  `sub-{participant}_ses-{session}_task-guessnumber_run-{run}_events.jsonl`
- 每个派生文件必须伴随同名 `.json` sidecar，记录来源、步骤、参数、版本、哈希。
- 连续 EEG 中间格式：FIF（MNE）或内存 `Raw`；禁止 CSV 作为唯一记录。

## 4. 通道与参考

- 默认物理绑定见 `config/channel_config.json`。
- 8 个 EEG 通道：Fz, Cz, P3, Pz, P4, PO7, PO8, Oz。
- **Cz 已占用为记录通道，因此不得使用说明书默认参考 Cz**；默认参考 A1（左耳垂），
  默认接地 Fpz。若实际帽子/贴片不同，必须修改配置并记录。
- 原始数据保留设备原始参考；后端的 CAR/平均重参考为派生步骤，必须记录原参考。

## 5. 前端技术方案

- 视觉刺激：Qt 全屏黑底白字，`stimulus_on_ms=200`，`blank_ms=1300`（SOA=1500ms，与文献 ISI 1500ms 对齐）。
- 每个 block：9 个数字 × R 次重复（默认 R=5，45 试次），随机但保证每个数字出现次数相等。
- 事件：每个刺激发送 `stim_on/{number}`、`stim_off/{number}` LSL marker，同时写
  EDF annotation 与 `events.jsonl`（记录 LSL 时间戳、单调时间、预期 onset）。
- 采集：500 Hz（推荐；后端可降采样到 250 Hz）；增益按 SDK 示例 Gain24；
  原始数据不进行任何软件滤波。在线显示滤波与离线滤波参数分开记录。
- 真实设备可用 SDK 的 EDF 录制；mock 模式用 pyedflib 生成 EDF+。

## 6. 后端与模型

- 信号处理核心：**ShallowConvNet**（Schirrmeister et al., 2017 结构），二分类 target/non-target。
- 9 选 1 决策：对每个试次输出 target 概率，按刺激数字聚合
  `mean(log(p/(1-p)))` 或平均概率，取最大值。
- 结构：temporal conv → spatial conv → BatchNorm → square → avg pool → log → dropout → 分类。
- 输入窗口：默认 `-200ms ~ +1000ms`（与文献 epoch 区间一致）。
- 预处理顺序：载入 EDF → 通道标签/类型校验 → 0.5Hz 高通 → 50Hz 陷波 →
  20Hz 低通（可配，默认）→ 降采样 → epoch → 基线校正 → xDAWN 可选增强通道 → 伪迹检测/标记 →
  训练集统计标准化。所有步骤带参数与原因。
- 训练：按 session 交叉验证；class-balanced BCE/Focal Loss；AdamW；
  早停（val balanced accuracy/AUC）；数据增强（时间抖动、幅度缩放、通道 dropout、噪声）。
- 集成：不同随机种子/折训练多个 ShallowConvNet，输出概率平均。
- 可选：温度缩放校准、每 block 动态停止。

## 7. 质量与日志

- 日志：Python `logging` + JSONL 事件日志；关键节点记录输入输出哈希。
- 报告：matplotlib 生成 ERP 图、试次拒绝图、混淆矩阵、AUC 曲线，归档到 reports。
