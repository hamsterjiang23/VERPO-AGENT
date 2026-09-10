# 模块职责

下列目录为实现位置，目前只记录接口职责，不包含伪装成可训练功能的空实现。

| 目录 | 待实现内容 |
|---|---|
| `environments/` | reset／step／终止／工具错误；可恢复状态与真实环境反馈 |
| `rollout/` | 多轮采样、旧策略 logprob、完整 history 和稳定 step/token ID |
| `replay/` | 全局反馈前缀、消融前缀、整轨迹 tokenization、预测位置对齐 |
| `objectives/` | 上游数学接口适配、action mask、独立 reference／evidence loss |
| `evaluation/` | 无特权提示的 Student 评测、成功率、工具成本、病态输出审计 |

只实现当前实验必要的接口；不增加 step gate 或收益预测网络。
