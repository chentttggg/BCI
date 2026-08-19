# Roadmap — 数据预处理与模型路线图

> 本路线图与 Constitution 3.x 一致，阶段编号沿用 Stage 0–5。
> 每个阶段的退出标准即质量门禁，见 `quality_gates.md`。

## Stage 0 — 数据接收与完整性校验

- 前端每次运行结束生成：EDF+、events.jsonl、session.json（采集元数据）、LSL 日志。
- 后端 `python -m backend.ingest`：
  - 校验文件存在、扩展名、大小；
  - 计算 SHA-256；
  - 写入 `data/manifest.jsonl`；
  - 建立 session 清单。
- 退出标准：所有文件通过完整性校验，原始区禁止修改。

## Stage 1 — 导入与标准化

- EDF → MNE `Raw`（或 pyedflib 兜底）→ 统一通道名、类型、单位 µV；
- 读取事件 marker，校验每个 block 9 个数字、重复数一致；
- 验证 LSL/EDF annotation 时间一致性，生成 `events_cleaned.tsv`；
- 写入派生副本 FIF 与 JSON sidecar。
- 退出标准：通道数=8、标签匹配配置、marker 无缺漏、采样率匹配、provenance 完整。

## Stage 2 — 初始质量检查与标注

- 指标：通道幅度、峰峰值、频谱、50Hz 线噪、导联脱落比例、坏段比例、事件一致性。
- 生成 QC HTML/PNG；标记问题，**不直接删除**。
- 退出标准：QC 报告完成，问题清单生成。

## Stage 3 — 基础信号处理

- 顺序（每步记录参数与原因）：
  1. 0.5 Hz 高通（去趋势，文献建议）；
  2. 50 Hz 陷波（工频）；
  3. 低通 20 Hz（P300 频带 + 抗混叠；干电极肌电抑制）；
  4. 降采样到模型采样率（默认 250 Hz，原始 500 Hz 保留）；
  5. epoch `-200 ~ +1000 ms`；基线校正 `-200 ~ 0 ms`；
  6. 可选 CAR 重参考（记录原参考与理由）。
- 退出标准：滤波前后频谱检查、事件 onset 重对齐验证、基线稳定。

## Stage 4 — 伪迹处理

- 伪迹检测（保守、可追溯）：
  - 每通道/trial 峰峰值阈值（默认 150µV）；
  - 绝对幅值阈值（默认 ±120µV，可配）；
  - 50Hz 残差检测；
  - 可选 ICA（保留成分权重与剔除理由）或 ASR。
- 输出：`artifact_mask.npy`、`artifact_report.json`；训练时剔除坏试次，
  推理时保留 mask 并在聚合中排除；坏导插值必须记录。
- 退出标准：伪迹残留低于阈值；坏导≤1；坏段比例不导致类别失衡。

## Stage 5 — 模型训练、最终 QC 与发布

- 交叉验证：默认 leave-one-session-out；单人单 session 时按 block 分折。
- 训练 ShallowConvNet 集成；输出 best weights、预处理参数、AUC/准确率报告。
- 推理：`python -m backend.predict`，输出逐试次概率、逐数字得分、最终猜测。
- 退出标准：达到 `mission/project_brief.md` 成功标准；所有派生文件有 provenance。
