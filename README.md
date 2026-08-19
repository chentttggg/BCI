# Guess Number P300 — 8通道干电极 ShallowConvNet 解码系统

复现 Moucek et al. (2017) “Guess the number” P300 oddball 实验，并扩展为
BrainSync BS8A 8 通道干电极（Fz, Cz, P3, Pz, P4, PO7, PO8, Oz）的
个人化 BCI 前端 + 后端系统。

> **重要**：先阅读 `guidance/` 指导文件夹与根目录 `Constitution.md`。
> 原始数据只读；所有处理从派生副本开始；质量门禁不可跳过。

## 目录

```text
guidance/              项目指导文件（mission/tech_stack/roadmap）
config/                通道、预处理、训练参数 JSON
frontend/              实验前端（刺激呈现 + BrainSync 采集 + EDF 记录 + LSL 打标）
backend/               ShallowConvNet 训练/推理/QC 后端
scripts/               合成数据生成器
data/raw/              原始 EDF + events.jsonl + session.json（只读）
data/derived/          派生数据（缓存、模型、报告、预测）
```

## 1. 安装

```bash
python -m venv .venv311
.venv311/Scripts/activate        # Windows；Linux/macOS 使用 source .venv311/bin/activate
pip install -r requirements.txt

# 仅真实设备采集需要安装 BrainSync SDK：
# pip install brainsync_sdk==0.3.0
```

## 2. 前端采集

```bash
# 真实设备，6 blocks x 5 次重复，受试者默想 7
python -m frontend.main --device --target 7 --blocks 6 --repetitions 5 \
    --output-dir data/raw --subject P01 --session 001

# 无硬件 mock 自测（短范式，便于快速验证）
python -m frontend.main --mock --headless --target 7 \
    --blocks 1 --repetitions 1 --stimulus-ms 20 --blank-ms 30 \
    --output-dir data/raw
```

前端默认参数：500 Hz、Gain24、刺激 200 ms、空白 1300 ms（SOA=1500 ms）、
每 block 9×5=45 试次。生成文件：

- `sub-..._eeg.edf`：未滤波原始 EEG（µV）
- `sub-..._eeg_events.jsonl`：每个 marker 的 LSL/monotonic/采样点索引
- `sub-..._eeg_session.json`：通道、REF/GND、目标数字、哈希、丢包统计

## 3. 合成训练数据（可复现 demo）

```bash
python -m scripts.make_synthetic_dataset --output-dir data/raw \
    --sessions 6 --blocks 2 --repetitions 5 --seed 100
```

## 4. 后端处理

```bash
# Stage 0：原始文件哈希入库与完整性检查
python -m backend.main ingest --data-dir data/raw --manifest data/manifest.jsonl

# Stage 2-4：QC 报告（PSD、ERP、伪迹 mask）
python -m backend.main report --data-dir data/raw --output-dir data/derived/reports

# Stage 5：训练 + 交叉验证 + 生产集成模型
python -m backend.main train --data-dir data/raw \
    --output-dir data/derived/models/guess_number --cv --production

# 推理：对一个 session 猜测默想数字
python -m backend.main predict \
    --edf data/raw/sub-P01_ses-001_task-guessnumber_run-001_eeg.edf \
    --model-dir data/derived/models/guess_number
```

训练核心为 Schirrmeister et al. (2017) 的 **ShallowConvNet**：
temporal conv → spatial conv → BatchNorm → square → avg pool → log →
dropout → 分类。先做 target/non-target 二分类，再按刺激数字聚合
`mean(logit)` 得到 1–9 的 9 选 1 猜测。默认使用多 seed 集成。

## 5. 主要改进（相对于原文献）

1. 3 通道 → 8 通道顶-枕覆盖；
2. 人工看平均 ERP → ShallowConvNet 单试次概率 + block 级聚合；
3. EDF/LSL/JSONL 三重事件记录，事件样本索引可追溯；
4. 保守伪迹阈值检测（只标记，不静默删除）；
5. 类别不平衡 focal loss、加权采样、时间/幅度/通道 dropout 增强；
6. 按 session 的 LOSO 交叉验证 + 多 seed 集成，降低单人小样本方差；
7. 算力优先级低：原始 500 Hz 保留，模型输入 250 Hz，集成不压缩。

## 6. 注意事项

- `config/channel_config.json` 必须与物理连线一致。Cz 是记录通道，
  不能按 BrainSync 说明书默认 Cz 作 REF；默认 REF=A1、GND=Fpz。
- 真实采集前先查阻抗：建议每个通道 < 5 kΩ，门槛 < 10 kΩ。
- 后端滤波顺序为：0.5 Hz 高通 → 50/100 Hz 陷波 → 40 Hz 低通 →
  250 Hz 降采样 → epoch(-0.2~1.0 s) → 基线(-0.2~0 s) → CAR。
  所有参数在 `config/preprocessing.json`。
- 重大参数修改必须记录到 `guidance/tech_stack/decisions_log.md`。
