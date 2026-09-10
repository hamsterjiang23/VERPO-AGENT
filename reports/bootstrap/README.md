# VERPO-AGENT 创建与依赖接入验证

日期：2026-09-10。范围：项目骨架、依赖来源、开发安装与文档同步；没有 Agent 训练或 GPU 性能结果。

## 完成内容

- 独立 GitHub 私有仓库 `hamsterjiang23/VERPO-AGENT`，默认分支 main。
- 固定 PGR-Probe 提交 `9c01f2bcd05aea1b3741ba60ede9aad201e0875c`，源码以 submodule 引用。
- 可安装的 `verpo_agent` 开发包、显式源码 runner、版本／源码指纹预检。
- 三份研究文档及来源 manifest；当前设计明确无 step gate、无收益预测器、无 FEC 投影。
- PGR-Probe 同步了新仓库入口和最新设计澄清。

## 已测范围

本机 macOS、Python 3.11.0：editable 开发安装成功；包版本为 0.1.0。

- 7 项单元测试通过，覆盖正常依赖、错误提交、脏依赖、gitlink 漂移、禁止的 gate、路径越界和缺失依赖。
- runner 解析到本仓库 `verpo_agent` 与锁定依赖的 `risk_aware_opsd`／`verl` 源码路径。这是源码定位检查，没有导入完整 GPU 训练栈。
- 默认系统 Python 3.9 被预检正确拒绝；使用独立 Python 3.11 开发环境通过。
- 结构化预检结果见 [preflight.json](preflight.json)，文档检查见 [documentation_validation.json](documentation_validation.json)。
- 从 GitHub SSH 全新克隆首个提交 `8575c28e2a718753137100a4e458b126b222b3e7`，按 README 初始化直接 submodule，27 项预检全部通过且工作区干净；没有使用本地源码路径替代远程依赖。结果见 [clean_clone_validation.json](clean_clone_validation.json)。

## 未完成项

环境 adapter、真实多轮 rollout、tokenizer 对齐实现、loss 接入、Ray worker 分发、生产 GPU runtime 锁定、模型／数据／评测协议和训练均待实现。不能根据此报告宣称可直接启动训练。

## 复现

```bash
.venv/bin/python scripts/preflight.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_with_upstream.py -- python -c "import verpo_agent; print(verpo_agent.__version__)"
```
