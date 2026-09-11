# 实现与运行合同

Schema 2 新增原生 ToolAgentLoop / BaseTool benchmark 适配，配置与资源部署见 [benchmarks.md](benchmarks.md)。下文 lookup/calculation 与 schema 1 的说明保留作核心回归合同；实际 benchmark 采用独立的环境服务、数据 manifest、每轮动作预算与评测温度。

## 训练语义

Student 生成多轮工具轨迹。训练时将真实工具结果、错误和最终成败放在 Teacher 前缀，再对整轨迹进行 causal teacher forcing。Student、冻结 reference 和无反馈 EMA 分支不接收这个前缀。

生成期间的 token IDs 是事实来源。环境文本直接追加为非策略 token，下一轮模型生成继续保留实际 token；不把完整文本重新 tokenize 来猜测 action 边界。原始 BOS 保留在最前面，反馈前缀插在 BOS 后，其余轨迹 token 完全保留。各分支保存 attention mask、position IDs、独立 predictor 索引与 target ID，并逐项核验；批量 padding 只在前向时添加。

环境的标准答案只用于判分；回放前缀采用字段白名单，不复制数据集私有 state、标准答案或整条原始 action。全轨迹反馈是 retrospective supervision，不声称继承旧推导的 predictable-direction 回报保证。

## 实验配置

`configs/agent_tools.template.json` 给出全部字段。除固定实现选择外，不提供旧实验默认值；训练模型和硬件尚未指定，因此模板不是一个已注册实验。

| 字段 | 语义 |
|---|---|
| `objective` | `residual_verpo` 或 `feedback_opd`，两个目标共享 reference 项 |
| `model.path` / `model.revision` | Hugging Face 模型 repo ID 与完整 40 位提交；按 revision 下载本地快照 |
| `runtime.package_versions` | 运行时精确匹配的 distribution 版本；`transfer_queue` 对应 `TransferQueue` |
| `dataset_manifest` / `output_dir` | 相对配置文件目录解析；数据文件校验 SHA256、数量和 split 唯一性 |
| `train_batch_size` / `group_size` | 每步任务数 / 每任务轨迹数；组大小至少 2 |
| `train_steps` | optimizer 更新预算；同步采样后一个完整 group batch 更新一次，PPO epochs=1 |
| `micro_batch_size_per_gpu` | 每 rank microbatch 轨迹数；整批轨迹数必须能按 GPU 数量和 microbatch 整除 |
| `lambda_ref` / `lambda_feedback` | 两组均需显式填写相同的正系数；初版 token 权重恒为 1 |
| `ema_decay` | 仅按成功 optimizer update 计数，范围 (0,1) |
| `max_prompt_tokens` / `max_response_tokens` | 原始 prompt 预算 / 包含环境文本的完整 response 预算 |
| `max_replay_tokens` | 包含反馈前缀的 Teacher 上下文预算，必须大于前两项之和且不超过模型容量 |
| `max_action_tokens` / `max_turns` / `timeout_seconds` | 每次生成预算、最大动作轮数、每次模型调用超时 |
| `validation_interval` / `checkpoint_interval` | 按训练 step 计数的评测与保存周期 |

显式实现默认：AdamW，betas=(0.9,0.999)，weight decay=0，恒定学习率，无 warmup，gradient clipping=1.0；标准 GRPO 使用组内 sample std 和 epsilon=1e-6；采样温度 1、top-p=1、top-k 禁用，评测为单次贪心。无额外 entropy reward、KL shaping、IS 筛选或 GRPO 组重采样。训练使用实际 rollout logprob 作为行为策略分母。

## veRL 接入与数据并行

采用固定 veRL 已展平的基础配置生成独立 native 配置；不调用 SDPO/RLCSD 等旧 launcher。Agent TaskRunner 创建自己的同步 Trainer；Trainer 选择 Agent worker。worker 通过独立 engine registry key 使用 Agent FSDP2 engine，HF 架构仍为 language model。

每个 microbatch 在 Student 前向前执行冻结 reference、EMA base、EMA evidence 三次批量前向。EMA 上下文退出并恢复 Student 参数后才构建 Student 图；action logits 保留至当前 loss 计算，随后释放。禁用 fused logits 与序列并行；首版采用 padded model forward，避免对模型的 packed attention 支持作额外假设。

复用 veRL 的 GRPO/PPO loss、TensorDict 重排/切分和全局 loss-mask 计数。蒸馏各 rank 使用 `local_sum × dp_size / global_action_count`，microbatch 损失求和，FSDP 梯度平均后等价于全局 token 均值。Teacher 的 logits 和 token 权重均停止梯度。

多节点需要显式 `RAY_ADDRESS`、相同虚拟环境、相同源码目录部署，以及所有节点可读写的同一运行目录和模型快照位置。每个 TaskRunner、worker 与 AgentLoop 会校验源码和包版本。初版不负责远程机器部署或 GPU runtime 安装，不把本地 PYTHONPATH 设置当成远端部署成功。

## 失败、恢复和评测

- 工具无效输入、工具错误、模型超时及预算终止保存在轨迹中；生成 backend 异常保存可用的部分轨迹并使采样失败。没有生成任何 action 的轨迹保存后报错，不伪造 action。
- 回放超长、target 错位、TQ token/mask 改变、过期策略或不完整 GRPO 组均拒绝更新。driver 保存回放诊断；worker 在 Teacher 前向前集体检查映射，失败时保存各 rank 的 `replay_errors/` 数据。失败的异步 Agent task 经自定义 replay buffer 传播，不作为完整组使用。
- checkpoint 保存 Student、optimizer/scheduler、RNG、EMA 分片与计数和 dataloader 状态。所有文件写完并生成哈希清单后才推进恢复指针；恢复核验完整性、源码、runtime、reference、实验参数和 world size。
- `evaluate` 使用最近完整 Student checkpoint 和 test split；初始状态无 checkpoint 时明确为 step 0。训练期间只使用 validation split，不选择或汇报 test 最优结果。
- 评测保留每条原始轨迹，输出成功率、工具调用数、无效调用率、截断率和超时数。CPU 工程测试不用于推断 Agent 性能。

运行记录位于 `run_manifest.json`、`native_training.yaml`、`raw_rollouts/`、`replay/`、`rollouts/`、`validation/`、`evaluation/` 和 `checkpoints/`。`replay-audit` 使用已有本地 tokenizer 检查单条轨迹，不发起模型下载。
