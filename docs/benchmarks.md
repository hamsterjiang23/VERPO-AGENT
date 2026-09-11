# AgentOPSD benchmark 与 veRL 适配

本仓库沿用 AgentOPSD 的 **ALFWorld、WebShop、Search-QA 环境与动作协议**，训练目标仍是本项目的 GRPO + residual VERPO / feedback OPD。官方论文与当前源码已经逐项核对；源码参考固定为 `ZethWang/AgentOPSD@0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b`，见 `benchmark_sources.lock.json`。不导入该仓库的 veRL 或算法实现。

来源：[论文 Appendix C/E/F](https://arxiv.org/html/2608.05987v1)、[官方环境实现](https://github.com/ZethWang/AgentOPSD/tree/0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b/agent_system/environments)、[官方启动脚本](https://github.com/ZethWang/AgentOPSD/tree/0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b/examples/agentopsd_trainer)。本地文献笔记中“代码尚未公开”的旧状态已不适用。

文献中的 benchmark 不是一套：[OCSD](https://arxiv.org/html/2608.04788v1) 和 [AHEAD](https://arxiv.org/html/2608.24114v1) 也使用上述三类环境；[TurnSight](https://arxiv.org/html/2608.04007v1) 在 FTRL 训练，评测 FTRL、BFCL、ToolHop，最多 10 轮。当前实现以 AgentOPSD 的三环境协议为范围，未实现 TurnSight 的函数工具执行器与评测器。

## 环境合同

| 环境 | 模型动作 | 环境与反馈 | horizon / 每轮生成 | 评测 |
|---|---|---|---|---|
| ALFWorld | `<think>…</think><action>command</action>` | 固定真实 `game.tw-pddl`；TextWorld 状态跨轮保留；返回 observation/admissible commands | 50 / 512 token | 六类任务分别统计，整体按样本加权成功率 |
| WebShop | `<think>…</think><action>search[query] 或 click[label]</action>` | 官方文本 WebAgentTextEnv；每条 rollout 独立 session；商品目录和索引共享 | 15 / 512 token | 原生部分完成 Score ×100，完全满足条件的成功率 |
| Search-QA | `<search>query</search>` / `<answer>answer</answer>` | Search-R1 HTTP 检索，E5/Wikipedia 2018；返回 `<information>`；答案仅在服务端判分 | 4 / 512 token | 答案别名规范化 EM；七个子集、micro 和 macro accuracy |

Search-QA 训练仅接受 NQ / HotpotQA，评测覆盖 NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle。报告明确记录实际覆盖的子集数，不能用单一子集结果声称七基准复现。

ALFWorld 默认划分为 train / valid_seen / valid_unseen，实际 game 文件与哈希逐条注册。WebShop 官方代码使用训练 goal index ≥500、评测 <500；论文报告 128 个固定评测任务。**这里要求显式记录选择的 indices**，不会擅自把任意 128 条声称为论文同一集合。验证与测试实例不可重复；若要比较论文固定集合，应将那个集合登记为最终评测，另取不重叠任务作开发验证。

训练 reward 统一为完全成功 1、否则 0。WebShop 部分分数单独记录，不混入 GRPO reward。官方 wrapper 的 10/0 奖励在这里缩放为 1/0；不继承其 invalid-action penalty、skill 检索、优势重塑、组筛选或其他算法超参数。

## 原生接入路径

`AgentTaskRunner → AgentLoopManagerTQ → AgentLoopWorkerTQ → BenchmarkAgentLoop(ToolAgentLoop) → BenchmarkEnvironmentTool(BaseTool) → HTTP 环境服务 → TransferQueue → AgentTrainer → AgentLoss`。

原生 veRL 负责 loop 注册、Ray worker 调度、生成调用、PENDING/GENERATING/PROCESSING_TOOLS 状态机、工具消息追加和队列后处理。适配层注册 benchmark 文本 parser，并将 BaseTool 的 create/release 生命周期扩展至整条轨迹，避免每个动作把环境重置。服务拒绝重复或乱序 turn；每条 GRPO 轨迹有独立环境 session，同任务组用相同初始任务和 seed。

只追加模型实际返回的 token IDs/logprobs；工具消息与人工模板 token 的 mask 为零，模型生成 EOS 保留。Student 图前的 Teacher 整轨迹回放沿用现有实现。采样、环境或映射异常保存原始数据并使采样组失败，不静默生成成功占位结果。

## 数据与环境部署

`configs/benchmarks/` 每种环境都有 profile、selection、service、experiment 四类 JSON。profile 中的环境 horizon 和动作格式可直接复用；其余模板中的 null 必须替换为实际资源和版本。

1. 用 `scripts/setup_benchmark_source.py --directory outputs/reference/AgentOPSD` 获取固定环境源码。只在独立环境服务进程加载其中的 ALFWorld/WebShop 包，训练仍使用固定 PGR-Probe。
2. ALFWorld 安装其 TextWorld 依赖并取得官方 PDDL/game 数据。WebShop 按官方 README 在独立 Python ≤3.10 环境安装，准备全部商品、属性和 Lucene index；服务进程不需要 vLLM。Search-QA 单独部署 E5 检索服务及 Wikipedia 索引，不把网页搜索或测试文档集合当作该检索库。
3. selection 指向真实 JSONL/parquet 源文件，填写 SHA256、数据 revision 与 selection_rule；可用 `indices` 明确选取行。Search-QA 输入支持 `question` + `answers` / `golden_answers` / `reward_model.ground_truth.target`。ALFWorld 行必须含 `gamefile`、`game_sha256`、`subset`；WebShop 行必须含 `goal_index`。
4. 运行 `prepare-benchmark` 生成有完整 split/hash 的 episode manifest。服务与训练配置使用同一 manifest；标准答案只在 episode 数据和服务端判分中使用，不放入 prompt。
5. service 配置填写资源路径和**实际安装版本**；resource manifest 形如 `{"provenance": {"revision": "实际版本"}, "files": {"相对资源路径": "SHA256"}}`。先 `check-environment` 校验，再启动服务。把 `/health` 返回的 identity 填入 experiment，供 driver 和每个 rollout 校验。

```bash
python -m verpo_agent prepare-benchmark --selection runs/search.selection.json --output datasets/search_registered
python -m verpo_agent check-environment --config runs/search.service.json
python -m verpo_agent serve-environment --config runs/search.service.json --port 8102
```

上述服务命令在各自的 benchmark Python 环境运行；安装本仓库包时无需安装训练 runtime。配置路径相对其 JSON 文件解析。服务默认只监听 `127.0.0.1`；多机训练需显式监听和配置所有 rollout worker 可达的地址，并在可信网络部署。资源不足、数据缺失、版本或服务 identity 不一致会报错。

```bash
.venv/bin/python scripts/run_with_upstream.py -- python -m verpo_agent validate-config --config runs/search.experiment.json
.venv/bin/python scripts/run_with_upstream.py -- python -m verpo_agent train --config runs/search.experiment.json
.venv/bin/python scripts/run_with_upstream.py -- python -m verpo_agent evaluate --config runs/search.experiment.json
```

## 与论文复现的边界

- 环境动作、终止规则和判分对齐；prompt 为本仓库编写，采用同样动作语法，不宣称逐字复刻论文提示词。
- 本项目保留完整 Student 因果历史；官方 AgentOPSD 有历史窗口和按轮 prompt 预算。这里的 `max_response_tokens` 是整条轨迹总预算，**不是**论文每轮的 512。`max_replay_tokens` 还须容纳反馈前缀。不能把 2048/4096/512 直接当成总上下文而悄悄截断动作。
- benchmark 配置显式采用官方环境 horizon、每轮 512 和可配置评测温度；不擅自复制论文模型、训练预算、学习率或蒸馏系数。旧 toy 模板只保留用于回归测试。
- CPU 原生链路测试使用 scripted LLM client 与小型 HTTP 检索 fixture，以检验实际 manager/worker/ToolAgentLoop/服务/队列接口；这不是 vLLM 模型生成，也不是完整 benchmark 数据评测。
- 当前机器尚未安装完整 ALFWorld/WebShop 数据与运行环境，也未部署全量 E5 检索索引；GPU/FSDP2/NCCL 和这些完整环境的部署验收仍需实际运行。不能据此声称论文成功率已复现。

## 真实资源登记补充

服务基础依赖用 `pip install -e '.[environment-service]'` 安装；ALFWorld/WebShop 引擎依赖按固定源码 README 另装。资源清单必须包含实际使用的文件：ALFWorld 配置（每条 game 另由 episode 哈希核对）；WebShop 商品、属性及全部 Lucene index 文件。Search-QA 的 `retriever_provenance` 必须登记 `model_revision`、`corpus_revision`、`index_sha256`，资源清单保留对应检索部署配置。服务检查不能证明远端检索器加载了声明的索引，目标部署仍须核验检索进程和实际资源。

提供真实资源目录转换器；缺少官方资源时会失败：

```bash
python scripts/catalog_alfworld.py --data-root datasets/alfworld --output datasets/alfworld_catalog
python scripts/catalog_webshop.py --source-root outputs/reference/AgentOPSD \
  --products-file datasets/webshop/items_shuffle.json --attributes-file datasets/webshop/items_ins_v2.json \
  --seed 17 --validation-count 128 --test-count 128 --output datasets/webshop_catalog
python -m verpo_agent prepare-benchmark --selection datasets/alfworld_catalog/selection.json --output datasets/alfworld_registered
```

WebShop catalog 在引擎依赖完整的服务环境运行：初始化商品目录读取真实 goal 总数，固定选取并保存 validation/test indices。ALFWorld catalog 登记官方 split 与 game，服务 reset 再检查游戏属于官方可解任务。Search-QA 可直接登记官方下载的 JSONL/parquet，显式选择七个评测子集；输出包含 ID/OOD micro EM，未覆盖的集合不会输出虚构零分。
