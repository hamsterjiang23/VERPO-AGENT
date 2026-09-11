# 固定上游依赖

PGR-Probe 是源码依赖，固定提交见[依赖锁](../upstream.lock.json)。本仓库不复制 veRL／VERPO 实现，不追随浮动 main。

## 为什么需要显式 runner

上游根 pyproject 当前只打包 `src*`；本地修改的 veRL 又导入 `risk_aware_opsd`。只安装上游根包或只安装 PyPI veRL 都不足以复现其运行代码。

`scripts/run_with_upstream.py` 先验证 gitlink、lock、checkout 和源码指纹一致，再在子进程中加入源码路径。GPU 训练依赖由具体实验的 runtime 单独锁定；`.[cpu]` 提供核心验证依赖，不包含 vLLM/CUDA runtime。CPU 安装成功不代表 GPU 训练环境验证通过。

不要调用上游旧 `pipeline/verl_math/run.sh` 来假装启动新的 Agent 协议。它有自己的已注册数据、配置与引擎语义。

## 克隆

```bash
git submodule update --init --depth 1 third_party/PGR-Probe
python3 scripts/preflight.py
```

不需要递归初始化上游内部无关参考 submodule。GitHub SSH 访问权限必须由使用者已有的登录提供；仓库不包含 token 或密钥。

## 显式升级

升级是一个需要评审的代码变更：检查上游改动 → fetch 指定提交 → checkout detached 指定 SHA → 更新 lock 的 commit 与关键源码指纹 → 审核并刷新研究快照及 provenance → 暂存 submodule gitlink → 运行预检及相关测试 → 提交。

不要使用 `git submodule update --remote` 自动改变实验代码。每次真实实验记录本仓库提交、上游完整 SHA、有效配置和 runtime manifest。

## 同步分工

- 通用 loss、Teacher、分布式兼容性修复：先在 PGR-Probe 开发和验证，再升级锁定版本。
- Agent 环境、feedback replay、评测与新协议：在本仓库维护。
- 本次没有迁移训练数据、模型、凭据、旧输出或大规模历史实验目录。
