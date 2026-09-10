# 当前设计与独立仓库边界

2026-09-10：从 PGR-Probe 的 Agent VERPO 讨论建立独立实验仓库。

## 已确认

- Student 按原始因果历史在环境中生成完整轨迹。
- 真实 observation／工具执行返回是相对早期 action 的训练时特权信息；已在历史内的观测不是该步新增特权。
- 将全局反馈放在 Teacher 输入前缀，整条轨迹 teacher-forcing。逐 step/token 得到分数不要求逐 step 前向。
- 只有策略生成目标参与损失，observation 保留为后续上下文。
- 当前不加 step gate、不加收益预测网络。mask 与门控不同。
- 第一阶段不做 FEC 投影；先研究反馈本身，再对照观测消融、原 token 控制器等组件。
- 所有 Teacher／Student 分支使用原 action 的显式 predictor 索引映射；不得假设前缀拼接后绝对位置相同。

## 仍需决定

初始环境、模型、轨迹长度与 batch、Teacher 更新方式、奖励定义、reference 和 evidence 系数、选定目标及验证集都未注册。配置文件中的 null 是未决定，不得被 launcher 自动替换成旧实验默认值。

绝对 feedback OPD 与 observation-residual VERPO 是两条不同训练目标。后者可复用上游概率差／log-density 的数学身份，但 Teacher 证据构造和回放接口需要新增。使用未来信息的目标不能直接继承原可预测方向理论保证。

## 下一阶段实现顺序

1. 固定一个环境、模型和可审计的小数据切片，注册训练与评测合同。
2. 保存真实多轮轨迹及所有工具失败，验证逐步历史可恢复。
3. 实现整轨迹反馈打包与 token 预测位置对齐，先做冻结模型回放核对。
4. 接入一个明确目标，验证 action mask、停止梯度、reference 身份和 loss reduction。
5. 在目标硬件锁定运行环境并检查显存，再进行明确授权的小规模训练。

## 收益网络讨论

可选收益网络预测“沿 Teacher 方向更新的预期增益”，常规 critic 则通常预测剩余回报。轻量 hidden-state 头可能只增加几百 MiB 缓存，但会引入噪声标签、校准和非平稳性；完整 critic 显存成本大得多。当前无实际收益证据，故不实现。完整估算和条件见[讨论稿](research/Agent_VERPO_讨论总结与Observation回放方案.md)。

## 文档同步

`docs/research` 是指定上游提交的阅读快照；文档中旧 baseline、R0/R1、FEC 与 step 控制均属理论或历史候选，不覆盖本页的当前决定。来源、源文件 SHA256 和转换后 SHA256 见 `docs/research/provenance.json`。后续 Agent 实验设计在本仓库维护，升级上游时显式审核快照变化。
