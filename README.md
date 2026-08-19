# Guess Number P300 — 8 通道干电极 ShallowConvNet 解码系统

复现 Moucek et al. (2017) “Guess the number” P300 oddball 实验，并扩展为
BrainSync BS8A 8 通道干电极（Fz, Cz, P3, Pz, P4, PO7, PO8, Oz）的个人化 BCI。

> 先阅读 `docs/guidance/` 与根目录 `Constitution.md`。
> 原始数据只读；所有处理从派生副本开始；质量门禁不可跳过。

## 项目结构（现代 src-layout）

```text
pyproject.toml              # 依赖、打包、CLI 入口、pytest/ruff 配置
src/guess_number/           # 源码包
  utils.py                  # 前后端共用：哈希 / 原子写入 / JSONL / 日志
  frontend/                 # 实验前端（轻量，随 exe 打包）
  backend/                  # 信号处理 / 训练 / 预测（由本地 Python 运行）
  gui/                      # 研究员图形界面（exe 入口）
  scripts/                  # CLI 脚本
  config/                   # 打包 JSON 配置
tests/                      # pytest 测试
docs/guidance/              # 项目指导文档
data/                       # 原始/派生数据（git 忽略）
```

## 安装

```bash
python -m venv .venv311
.venv311/Scripts/activate        # Windows；Linux/macOS 使用 source .venv311/bin/activate
pip install -e .[dev,backend]   # backend 为 MNE/torch 等离线处理依赖
```

安装后获得命令：

```text
guess-number-frontend
guess-number-backend
guess-number-check-edf
guess-number-make-synthetic
guess-number-tap-test
guess-number-researcher      # 研究员图形界面
```

## 研究员图形界面 / EXE

源码入口：`src/guess_number/gui/researcher.py`。打包命令：

```bash
pyinstaller --noconfirm --clean packaging/guess_number_researcher.spec
```

产物：

```text
dist/GuessNumberResearcher.exe
```

图形界面把实验步骤和常用后端步骤都做成了按钮：

- 检查设备（USB / 蓝牙）
- 填写受试者/目标数字/blocks/重复次数/采样率/增益/时序
- 开始实验 / 停止实验
- 实验结束后弹窗记录受试者猜的数字
- 完整性检查、QC 报告、训练、预测数字

### exe 定位：轻量图形化 + 接口

`GuessNumberResearcher.exe` 只打包 PySide6、numpy、pyedflib、pylsl 与
BrainSync SDK，负责图形界面、刺激呈现和实时采集。MNE、PyTorch、SciPy、
pandas、scikit-learn、matplotlib **不打包进 exe**；点“完整性检查 / QC /
训练 / 预测”时，exe 调用“数据后端”页中选择的本地 Python 解释器：

```text
python -m guess_number.backend.main <ingest|report|train|predict> ...
```

Python 解释器默认自动查找：`GUESS_NUMBER_PYTHON` 环境变量 → 项目
`.venv311/Scripts/python.exe` → PATH 中的 `python`。若移动了环境，可在
GUI“数据后端”页重新选择。当前 exe 由 253 MB 降至约 57 MB。

### 刺激窗口：默认窗口模式，可拖到第二块屏幕

实验页新增“刺激显示”：
- **窗口（可拖到扩展屏第二屏）**：默认。黑色刺激窗口不再是全屏，研究者可
  把它拖到第二块显示器上给受试者看，主屏保留研究员控制台。
- **全屏**：保留原来的全屏黑底模式。
- **目标屏幕**：选择扩展桌面的目标显示器，窗口/全屏都会优先放在该屏。
- **预览/定位刺激窗口**：开始实验前先开一个预览窗，摆好位置再开始。

按 `Esc` 或关闭刺激窗口都会停止实验。

## 前端采集

```bash
# USB 真实设备，6 blocks × 5 次重复
guess-number-frontend --device --target 7 --blocks 6 --repetitions 5     --output-dir data/recordings --subject P01 --session 001

# 蓝牙设备（与 USB 一样固定 250 Hz）
guess-number-frontend --ble --ble-name BrainSync --sfreq 250     --target 7 --blocks 6 --repetitions 5     --output-dir data/recordings --subject P01 --session 001

# mock 自测
guess-number-frontend --mock --headless --target 7     --blocks 1 --repetitions 1 --stimulus-ms 20 --blank-ms 30     --output-dir data/recordings
```

默认：250 Hz、Gain24、刺激 200 ms、空白 1300 ms。点击开始实验即开始采集，
第一个数字前默认 2 秒黑屏静息基线（`--baseline-black-ms` 可调）。

结束自动保存到 `data/recordings/<开始时间>/`：

```text
总 EDF
eeg_block_000.edf ...
events.jsonl
session.json
experiment_summary.json
split_manifest.json
```

数字刺激通过 EDF annotation + events.jsonl + LSL marker 三路记录，并用
LSL↔EEG 线性拟合自动对齐时间戳。GUI 结束会弹窗记录受试者猜的数字；
headless 用 `--subject-guess N`。

## 后端处理

```bash
# 完整性检查 + 对齐验证
guess-number-backend ingest --data-dir data/raw --manifest data/manifest.jsonl

# QC 报告
guess-number-backend report --data-dir data/raw --output-dir data/derived/reports

# 训练
guess-number-backend train --data-dir data/raw     --output-dir data/derived/models/guess_number --cv --production

# 预测
guess-number-backend predict     --edf data/raw/sub-P01_ses-001_task-guessnumber_run-001_eeg.edf     --model-dir data/derived/models/guess_number
```

## 工作台记录路径

工作站 `multimodal_hub` 的默认 EDF 保存目录已指向项目根目录下的 `Data/`。
录制时 SDK 会在该目录生成 `streaming_EEG_*.edf` 文件。

## 常用脚本

```bash
# 检查一份 EDF 的通道状态
guess-number-check-edf <edf文件> --plot data/derived/reports/qc.png

# 生成合成训练数据
guess-number-make-synthetic --output-dir data/raw --sessions 6 --seed 100

# 电极逐通道敲击测试
guess-number-tap-test --seconds 20 --gain Gain24
```

## 模型要点

- 主干：Schirrmeister et al. (2017) ShallowConvNet；
- 输入：原始 8 通道 + xDAWN target 2 成分 + non-target 1 成分；
- 预处理：0.5–20 Hz 带通、50/100 Hz 陷波、CAR、保守伪迹标记；模型与原始 EDF 均固定 250 Hz；
- 训练：focal loss、类别平衡采样、mixup、受控通道 dropout、多 seed 集成；
- 聚合：逐试次 P(target) → mean(logit) → 9 选 1，输出 block 置信度。

## 配置

打包配置在 `src/guess_number/config/`：

```text
channel_config.json            # 通道顺序、REF/GND
preprocessing.json             # 滤波、xDAWN、QC 阈值
train.json                     # 模型与训练参数
```

可用环境变量 `GUESS_NUMBER_CONFIG_DIR` 指定外部配置目录。

## 测试

```bash
pytest
```
