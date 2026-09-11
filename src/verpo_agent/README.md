# 核心模块

- `trajectory.py` / `rollout.py`：实际采样 token、行为 logprob、工具观测和终止原因；异步多轮采样。
- `environment.py` / `data.py`：受限计算器、查询环境、带校验和的独立数据 split。
- `replay.py`：训练时反馈前缀、完整轨迹回放、各分支显式 predictor 索引。
- `objectives.py`：复用固定 PGR-Probe 的 Fixed FKL residual loss，提供 OPD 对照和全局 token 归一化。
- `verl_ext/`：原生 veRL TaskRunner、Trainer、worker、FSDP2 engine、AgentLoop、失败传播和 loss 接口。
- `config.py` / `native.py` / `provenance.py`：独立配置、native 投影、来源校验和启动；训练命令不会模拟成功。
- `evaluation.py`：从原始 Student 轨迹汇总评测指标。

原生接口、CPU 参数更新、Gloo 和 Ray/TransferQueue 验证不代表 GPU/FSDP2/NCCL 端到端训练通过。验证范围见 `reports/core_implementation/`。
