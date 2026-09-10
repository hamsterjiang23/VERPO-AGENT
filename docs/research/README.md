# 研究文档阅读入口

| 文档 | 内容 |
|---|---|
| [完整推导](VERPO到Agent_VERPO_完整推导.md) | 原 VERPO → Agent、Notation、最优权重、训练伪代码 |
| [讨论总结与 Observation 回放方案](Agent_VERPO_讨论总结与Observation回放方案.md) | 文献对照、全局反馈前置、整轨迹回放、无 FEC 候选目标、收益网络显存与取舍 |
| [原联合控制报告](Agent_VERPO_联合控制研究与实现报告.md) | 联合控制理论和历史 R0/R1 候选，非当前训练配置 |

这些是上游固定提交的同步快照，来源和 SHA256 见 [provenance.json](provenance.json)。原稿中旧的 step gate、收益预测器和 FEC 方案保留作理论背景；当前实现范围以[当前设计](../current_design.md)为准。

显式更新上游 lock 后运行 `python scripts/sync_research.py` 刷新快照，然后检查差异和公式。预检会发现未同步的源／导出文件指纹。
