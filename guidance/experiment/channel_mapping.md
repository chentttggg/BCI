# 通道映射与参考配置

## 默认绑定（config/channel_config.json）

| 硬件 Channel | 电极标签 | 说明 |
| --- | --- | --- |
| 0 | Fz | 额中线 |
| 1 | Cz | 中央中线 |
| 2 | P3 | 左顶 |
| 3 | Pz | 顶中线（P300 核心） |
| 4 | P4 | 右顶 |
| 5 | PO7 | 左顶枕 |
| 6 | PO8 | 右顶枕 |
| 7 | Oz | 枕中线 |

- REF：A1（耳部电极）。
- GND：A1（本套电极帽 A1 为耳部接地/REF 共用电极，`ref_gnd_combined=true`）。
- 不使用说明书默认的 Cz REF 或 Fpz/AFz GND。

## 重要规则

- 物理连线决定信号来源；JSON 标签只负责赋予通道生物学意义。
- A1 不在 8 个 EEG 记录通道内；本配置允许 REF=GND=A1（共用）。
- 因为 Cz 是记录通道，**不能**使用说明书默认 REF=Cz 方案。
- 每次实验前打印/核对映射，并把最终 ref/gnd 写入 session.json。

## BrainSync SDK ChannelConfig

- 程序实际使用的配置源：`config/channel_config.json`。
- 前端启动时会显式构造 `brainsync_sdk.ChannelConfig`，labels/ref/gnd 与上表一致，
  避免使用 SDK 默认的 C6/C4/FC4/... Montage。
- BrainSync GUI / Multimodal-Hub 兼容格式副本：
  `config/brainsync_gui_channel_config.json`（仅用于图形工具导入，后端不读取）。
