# VERPO-AGENT

用于多轮 Agent on-policy self-distillation 的独立实验仓库。公共 veRL／VERPO 实现来自固定版本的 [PGR-Probe](https://github.com/hamsterjiang23/PGR-Probe)，Agent 环境、回放和实验在这里独立开发。

**当前交付是项目骨架、依赖接入和研究文档；还没有可运行的 Agent 训练器，也没有启动实验。**

## 当前方案

```text
Student 在环境中生成完整轨迹
    → 收集 observation、工具返回与终局反馈
    → 将全局反馈前置，Teacher 整轨迹回放
    → 对齐原 action token 的预测位置
    → 在 Student 原始因果上下文上计算 action-only 训练损失
```

不加 step gate，不加收益预测网络，初始方向构造不使用 FEC 投影。是否使用观测消融分支，以及绝对 OPD／残差纠正目标，见[设计合同](configs/observation_replay_design.json)和[当前决策](docs/current_design.md)。

## 获取与本地预检

需要 Python 3.10+、Git，以及对两个 GitHub 仓库的读取权限。默认依赖 URL 使用 SSH；沿用已经配置的 GitHub SSH 登录。

```bash
git clone git@github.com:hamsterjiang23/VERPO-AGENT.git
cd VERPO-AGENT
git submodule update --init --depth 1 third_party/PGR-Probe
python3 scripts/preflight.py
bash scripts/setup_dev.sh
.venv/bin/python -m unittest discover -s tests -v
```

只初始化直接依赖即可，不需要递归下载 PGR-Probe 内无关的历史参考仓库。`setup_dev.sh` 只创建本仓库 `.venv` 并安装轻量开发包，不安装 CUDA／PyTorch／vLLM，也不下载模型和数据。

如果默认 `python3` 低于 3.10，请用 Python 3.10+ 执行预检，并通过 `VERPO_AGENT_PYTHON` 指定开发环境解释器，例如 `VERPO_AGENT_PYTHON=python3.11 bash scripts/setup_dev.sh`。

## 引用公共代码

```bash
.venv/bin/python scripts/run_with_upstream.py -- python -c "import verpo_agent; print(verpo_agent.__version__)"
```

runner 在校验依赖后，将当前环境的 Python 放在 PATH 前面，并将本仓库 `src`、固定依赖中的 `verl` 根目录和 PGR-Probe 根目录加入子进程的 PYTHONPATH。它不修改全局 Python 环境，不会自动运行旧 launcher。

这个接入解决源码定位，不替代生产依赖安装。实际导入训练损失前还需要 PyTorch 等依赖；Ray 远程 worker 的工作目录和 runtime_env 分发尚待接入。生产 GPU 环境锁定属于下一阶段，不能把开发环境安装成功当成 GPU 训练验证。

## 项目入口

| 路径 | 用途 |
|---|---|
| [当前决策](docs/current_design.md) | 回放语义、模块边界、下一步实现顺序 |
| [研究文档](docs/research/README.md) | 完整推导、Notation、论文对照、训练伪代码及收益网络讨论 |
| [上游依赖说明](docs/upstream.md) | 固定版本、接入方式与更新步骤 |
| [依赖锁](upstream.lock.json) | PGR-Probe 精确提交与关键源码指纹 |
| [设计合同](configs/observation_replay_design.json) | 非运行配置，明确禁用 step gate／收益网络／FEC |
| [预检](scripts/preflight.py) | 依赖版本、工作区、源码与文档快照检查 |
| [模块职责](src/verpo_agent/README.md) | 环境、rollout、replay、objective、evaluation 的实现边界 |
| [验证报告](reports/bootstrap/README.md) | 此次仓库创建的验证结果与未完成项 |

公共代码修复先回到 PGR-Probe，再显式升级本仓库依赖；Agent 专属实现留在这里。版本升级不会自动发生。第三方代码保留自身许可与来源，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
