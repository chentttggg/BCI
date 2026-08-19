# 猜数字 P300 实验 — 指导文件夹

> 本文件夹是本次 EEG 项目的最高指导文件集合，依据根目录 `Constitution.md` 建立。
> 效力顺序：`Constitution.md` > 本文件夹各文件 > 代码内注释/README。
> 子文件只能细化宪法，不得放宽或违反宪法中的红线与原则。

## 目录说明

| 路径 | 内容 |
| --- | --- |
| `guidance/mission/` | 项目使命、不可妥协原则、本次实验边界 |
| `guidance/tech_stack/` | 技术栈、数据标准、模型选型、重大技术调整记录 |
| `guidance/roadmap/` | 处理阶段路线图、质量门禁、里程碑 |
| `guidance/experiment/` | 实验范式、通道绑定、操作检查清单 |
| `guidance/references/` | 文献要点与本次改进说明 |

## 项目一句话定义

复现 Scientific Data (Moucek et al., 2017) 的 “Guess the number” P300 oddball 实验：
受试者默想 1–9 中的一个数字，视觉随机呈现 1–9；系统根据单试次 ERP 与
ShallowConvNet 模型自动猜测受试者所想的数字。本机使用 BrainSync BS8A
8 通道干电极帽（Fz, Cz, P3, Pz, P4, PO7, PO8, Oz），单人多次测试，建立个人化模型。

## 执行顺序（必须遵守）

1. 先阅读本指导文件夹，再运行任何程序。
2. 先检查 `src/guess_number/config/channel_config.json` 与物理连线一致，再开始实验。
3. 前端只保存原始数据（EDF+ 与不可变事件日志）；滤波/伪迹处理只发生在派生副本。
4. 后端任何处理必须通过 `guidance/roadmap/quality_gates.md` 定义的质量门禁后才可进入下一阶段。
5. 模型重大参数变更必须写入 `guidance/tech_stack/decisions_log.md`。

## 版本

- 指导文件夹版本：0.1.0（初始建立）
- 状态：活文档；修改需在 `guidance/tech_stack/decisions_log.md` 记录原因。
