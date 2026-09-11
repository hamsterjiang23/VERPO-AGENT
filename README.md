# VERPO-AGENT

多轮 Agent 的反馈回放与 GRPO 蒸馏实验仓库。Student 执行工具任务后，EMA Teacher 带着真实执行反馈回放完整轨迹；训练仍使用 Student 原来的因果历史。

**已接入 veRL 原生 `ToolAgentLoop` / `BaseTool`，提供 ALFWorld、WebShop、Search-QA 环境适配与配置。见[benchmark 部署](docs/benchmarks.md)和[适配验证报告](reports/benchmark_adaptation/README.md)。CPU 接口验证不代表全量 benchmark 或 GPU 验收；未启动正式训练。**

## 两个目标

两组共享 GRPO 和冻结初始 reference：

- 残差 VERPO：`GRPO + λ_ref KL(q_ref || p) + λ_feedback [-Σ(q_e − q_0) log p]`。
- 反馈 OPD：`GRPO + λ_ref KL(q_ref || p) + λ_feedback KL(q_e || p)`。

`q_0` 与 `q_e` 使用同一版本 EMA Teacher，区别是有无全局反馈前缀；`q_ref` 不随 EMA 更新。固定 token 权重为 1，不启用 step gate、组筛选、收益网络、FEC 或额外 KL shaping。全同奖励组的 GRPO 优势为零，但仍参与蒸馏。

## 本地验证

需要 Python 3.10+，已验证环境为 Python 3.11 / PyTorch 2.8。安装范围为 CPU 测试依赖，不包含 GPU/vLLM runtime。

```bash
git submodule update --init third_party/PGR-Probe
bash scripts/setup_dev.sh
uv pip install --python .venv/bin/python -e '.[cpu]'
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/run_with_upstream.py -- python scripts/verify_core.py --output outputs/benchmark_validation --report reports/benchmark_adaptation/validation.json
```

验证脚本运行完整 unittest、真实本地 Ray/TransferQueue 和双进程 Gloo，并保存随机小模型的实际 optimizer 更新、EMA 状态及回放记录。脚本化工具轨迹用于工程验证，不作为 Agent 能力结果。

## 数据与运行入口

```bash
.venv/bin/python scripts/run_with_upstream.py -- python -m verpo_agent generate-data \
  --output datasets/tools_v1 --seed 17 --train 128 --validation 32 --test 32
.venv/bin/python scripts/run_with_upstream.py -- python -m verpo_agent --help
```

上述生成器是 lookup/calculation 回归 fixture。实际 benchmark 使用 [configs/benchmarks](configs/benchmarks)，按[部署流程](docs/benchmarks.md)注册真实资源。旧 toy 配置保留用于回归验证。

将[配置模板](configs/agent_tools.template.json)复制到自己的运行目录，显式填写模型完整 revision、runtime、数据 manifest、GPU 数量及实验超参数。模板中的 null 必须填写，不是可启动实验。

```bash
.venv/bin/python scripts/run_with_upstream.py -- python -m verpo_agent validate-config --config runs/registered.json
.venv/bin/python scripts/run_with_upstream.py -- python -m verpo_agent train --config runs/registered.json
.venv/bin/python scripts/run_with_upstream.py -- python -m verpo_agent evaluate --config runs/registered.json
```

`validate-config` 不分配 GPU；`train` 实际进入 veRL 训练，需先安装并匹配已注册 CUDA runtime；`evaluate` 恢复当前运行最近的完整 checkpoint，使用独立 test split。尚无 checkpoint 时评测初始 Student，并记录 step 0。以上命令只是使用说明，本次没有执行正式训练。

首版支持文本因果模型、FSDP2、vLLM、全参数更新、SP=1、非 fused full logits；训练采样温度为 1，benchmark 评测温度显式配置。模型须有可用的 tokenizer/chat template，且上下文容量覆盖 Teacher 回放预算。benchmark 使用注册于 ToolAgentLoop 的动作 parser 与跨轮持久 BaseTool 会话。详细参数与部署限制见[运行合同](docs/implementation.md)。

## 来源与记录

PGR-Probe 使用 Git 子模块固定快照。2026-09-11 实施前重新核对，远端 main 的最新 SHA 为 `9c01f2bcd05aea1b3741ba60ede9aad201e0875c`，与当前固定版本一致。本次扩充关键接口源码指纹，未修改第三方源码。

- [当前设计](docs/current_design.md)：回放语义及模块边界。
- [依赖接入](docs/upstream.md)：源码定位、快照升级和许可证。
- [研究文档](docs/research/README.md)：历史推导快照，不替代当前实现合同。
- [实现与验证报告](reports/core_implementation/README.md)：验证结论和局限。

原始轨迹、工具错误、replay 输入、loss 和 checkpoint 保存到显式运行目录。大文件与原始数据留在 `outputs/`、`runs/`、`datasets/` 或 `checkpoints/`，不提交 Git。
