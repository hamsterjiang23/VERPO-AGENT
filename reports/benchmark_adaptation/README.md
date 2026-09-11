# 原生 veRL benchmark 适配验证（2026-09-11）

实现基于固定 PGR-Probe 的原生 ToolAgentLoop 状态机和 BaseTool，新增 ALFWorld、WebShop、Search-QA 的真实引擎接口、跨轮环境会话服务、数据/资源登记与评测指标。配置及论文来源见 [benchmark 文档](../../docs/benchmarks.md)。旧核心验证报告保留为 v0.2 历史证据。

## 已验证

[机器可读报告](validation.json)：28 项测试全部通过，无跳过。依赖预检通过，PGR-Probe 仍为 `9c01f2bcd05aea1b3741ba60ede9aad201e0875c`，源码无修改。外部环境接口参考 `AgentOPSD@0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b`，已核对锁定源码指纹。

新增集成测试实际启动 `AgentLoopManagerTQ → Ray AgentLoopWorkerTQ → BenchmarkAgentLoop(ToolAgentLoop) → BenchmarkEnvironmentTool(BaseTool) → FastAPI SessionService → SearchEpisode → HTTP 检索 fixture`，并经真实 TransferQueue / ReplayBuffer 送回两条独立会话轨迹，再校验 Teacher/Student replay token 映射。原生 manager 与 worker 没有用替身替换。LLM generate client 是脚本替身，检索内容是小型 fixture。

其他验证覆盖数据 hash/split、动作投影/EM、会话隔离、乱序拒绝、资源 provenance、超轮数截断、真实 CPU optimizer 与 EMA、异常恢复、checkpoint、双进程 Gloo 全局归一化。发现并修复旧配置中 ReplayBuffer 不接受 `max_off_policy_threshold=0` 的问题，改为合法值 1；Trainer 与 benchmark loop 仍校验实际策略版本。

两种目标的随机 CPU 模型参数变化 L2 均约 0.052505，EMA 成功更新各 1 次。这是工程验收，不是模型训练收益。完整原始轨迹、replay、检索请求、测试日志、CPU checkpoint/EMA 保存在忽略目录 `outputs/benchmark_validation/`；原生链路证据位于其中的 `native_benchmark/`。

## 未验证与复现边界

- 尚未部署全量 ALFWorld 游戏运行环境、WebShop 商品/索引及 E5/Wikipedia 检索资源，因此不宣称真实完整 benchmark episode 已在本机运行。
- GPU/vLLM/FSDP2/NCCL 分布式训练及生产模型端到端生成未验收，未启动正式训练，无 benchmark 成功率结论。
- 当前 benchmark 范围为 AgentOPSD 的三环境；TurnSight 的 FTRL/BFCL/ToolHop 未实现。
- 动作协议与环境 API 参照论文及官方源码，但 prompt 由本仓库编写，保留完整因果历史，采用独立显式 split、1/0 reward 和本项目 VERPO/OPD 目标；不声称论文实验逐项复现。
- 本机 Ray 有资源关闭和平台弃用警告，未造成测试失败；CPU 测试不能替代目标 Linux/GPU runtime 验收。

复验命令：

```bash
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/run_with_upstream.py -- python scripts/verify_core.py \
  --output outputs/benchmark_validation --report reports/benchmark_adaptation/validation.json
```

报告记录验收时基线提交与源码/测试 SHA256。验收发生在提交前，agent_commit 是当时的基线；以报告内文件指纹识别本次被验证内容。
