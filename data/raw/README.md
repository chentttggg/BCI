# 原始数据区（只读）

- 前端程序只会在这里**新建**文件，不会覆盖或删除已有文件。
- 后端程序只读取本目录，不做任何修改。
- 生成文件：
  - `sub-*_eeg.edf`：未滤波原始 EEG；
  - `sub-*_eeg_events.jsonl`：事件/标记日志；
  - `sub-*_eeg_session.json`：session 元数据与哈希；
  - `*.edf.sidecar.json`：EDF provenance。
- 入库信息见 `../manifest.jsonl`。
