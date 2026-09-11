# Agent VERPO 核心实现与 CPU 验证

2026-09-11。首版核心功能和 veRL 分布式扩展已落地，完整验收结果见 [validation.json](validation.json)。**GPU/FSDP2/NCCL 端到端路径尚未验证，没有启动正式训练，也没有 Agent 基准性能结论。**

## 已实现

- 确定性 lookup/calculation 环境、二元判分、独立 train/validation/test manifest。
- 多轮实际 token 轨迹、动作 mask、行为 logprob、工具返回、错误及终止原因。
- 带全局反馈的整轨迹 Teacher replay，Student/reference/base/evidence 四分支显式 predictor 映射。
- 共享 GRPO 和冻结初始 reference 的 residual VERPO / feedback OPD 两个目标，EMA Teacher 与固定 token 权重。
- 独立 veRL TaskRunner、同步 Trainer、worker、FSDP2 engine、AgentLoop、失败传播与运行配置。
- checkpoint 完整性清单、EMA 状态、实验来源及 Student 评测记录。

## 验证结果

23 项测试通过，0 失败、0 错误、0 跳过。运行版本：Python 3.11.0、PyTorch 2.8.0、Transformers 4.56.2、Ray 2.48.0、TensorDict 0.10.0、TransferQueue 0.1.8；系统为 macOS CPU。完整版本和源码文件哈希写入机器可读报告。

| 检查 | 实际证据 |
|---|---|
| 数学身份 | 两个目标与直接 PyTorch 公式的 loss/gradient 对照；相同反馈分布时残差为零 |
| 对齐和 mask | 多轮工具、EOS、padding、变长前缀、超长拒绝、序列化、重排和 microbatch 检查 |
| 真实参数更新 | 随机小型因果模型完成 backward/optimizer step，验证非数值噪声的梯度与参数变化 |
| EMA/恢复 | 成功更新计数、跳过更新、Teacher 异常后恢复 Student、EMA 存取、optimizer 恢复连续性 |
| veRL 接口 | 实际导入原生扩展，构造冻结 dataclass 配置，调用原生两阶段 logits/loss 与 PPO loss |
| 数据并行 | 两个真实 Gloo 进程，2 对 5 个 action token、不等 microbatch，聚合 loss/gradient 与单批 oracle 一致 |
| 数据传输 | 实际启动本地 CPU Ray 和 TransferQueue SimpleStorage，回放字段往返、逆序读取和切分保持一致；调用真实 Trainer 的行为 logprob、优势、回放打包及 worker 数据桥接路径 |
| 记录与评测 | 通过真实队列读取轨迹并验证原始 JSONL、成功率和评测汇总；GPU worker RPC 与采样端由 CPU harness 替代 |
| 来源 | 最新远端快照核对、gitlink/lock/checkout 和关键源码指纹预检 |

CPU 持久化演示包含 86 个策略 token，每组各做一次更新，测试系数均为 1，EMA decay=0.9、学习率=0.001。这些数值只用于工程测试，不是注册的正式实验参数。

| 目标 | 反馈 loss | 梯度 L2 | 参数变化 L2 | EMA 更新次数 |
|---|---:|---:|---:|---:|
| residual VERPO | -0.000126863 | 0.000957411 | 0.0525050 | 1 |
| feedback OPD | 0.0000234315 | 0.000957412 | 0.0525049 | 1 |

残差 loss 是有符号纠正项，负值符合定义。初始 Student、base 和 reference 相同，因此两种反馈目标的首步梯度相同；这不说明后续训练效果相同。

## 可复查产物

本地输出保存在被 Git 忽略的 `outputs/core_validation/`：

- `tests.log`：完整测试结果。
- `trajectory.json`：脚本执行的真实本地工具轨迹及 token/mask/logprob 记录。
- `replay.json`：四分支完整输入与 predictor/target 映射。
- `metrics.json`：两次实际 CPU 参数更新的数值。
- `residual_verpo/`、`feedback_opd/`：随机模型/optimizer checkpoint 和上游 EMA sidecar。

工具调用序列由脚本提供，行为 logprob 是测试占位数据，未作为真实模型采样证据或性能结果使用。持久化演示验证的是回放与更新核；原生 GRPO/PPO 接口另由测试覆盖。

## 限制及下一步

模型、GPU 配置及具体实验超参数仍未注册。尚未验证 CUDA 上的 FSDP2 参数布局、NCCL、vLLM 连续调用、显存峰值、跨节点共享路径和分布式 checkpoint 恢复。需要在指定硬件完成这些检查后才能开展正式比较。

本次未修改第三方源码。实现前远端 main 与固定子模块均为 `9c01f2bcd05aea1b3741ba60ede9aad201e0875c`，依赖锁新增了此次使用的 trainer/worker/EMA/TensorDict 接口指纹。报告记录的是验证时的工作树文件哈希；`agent_commit` 是实施前基线提交，应结合源码和测试文件哈希定位本次验证的实现。

复现入口：`python scripts/run_with_upstream.py -- python scripts/verify_core.py`。CPU 依赖见项目 `.[cpu]` 和 `requirements-cpu.lock.txt`，不能用该 CPU 环境替代具体 GPU runtime。
