# Agent VERPO：讨论总结、Observation 特权信息与整轨迹回放方案

> 上游研究快照，来源提交 `9c01f2bcd05aea1b3741ba60ede9aad201e0875c`。历史候选中的 FEC、step gate 或收益预测器不覆盖[当前设计](../current_design.md)。原文与导出文件指纹见 [provenance.json](provenance.json)。

日期：2026-09-10。本文整理本轮讨论形成的理解、文献依据和候选训练方案。它是研究设计，不是已接入训练器的实现，也不注册或启动新的训练协议。

## 1. 阅读入口与当前结论

完整的旧 VERPO → Agent VERPO 数学链条、Notation 表格及训练伪代码见[完整推导](VERPO%E5%88%B0Agent_VERPO_%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md)。本篇承接其后，说明为什么可以重新选择 Agent 的证据来源，以及怎样在自回归模型中实现利用未来反馈的回放。

目前明确的结论：

1. 原 VERPO 已经有 token 权重，Agent 扩展继承“候选纠正方向是否支持任务收益、应接受多少”的思想。
2. 对角二次近似下的收益／代价比，与原 VERPO 的二次推导同构；这不代表实际平滑控制器就是该二次目标的精确最优解。
3. FEC 是证据方向的一种构造方式，并非 Agent VERPO 的必要组成。Agent 可以直接利用自身工具执行返回和后续 observation。
4. 自回归 Teacher 可以利用未来反馈：将反馈放进回放前缀即可，不必改成 BERT。
5. 当前确认的第一版回放形式是**全局反馈前置、整轨迹 teacher-forcing、只在策略生成 token 上计算损失**。逐 step 得到分数不等于逐 step 调用模型。
6. Observation 残差、全局复盘、step 选择和最优权重是不同模块，应分开验证。关于收益的理论保证不能随证据来源改变而自动继承。

## 2. 原 VERPO 与 Agent VERPO：到底相同和不同在哪里

| 维度 | 原序列 response VERPO | Agent VERPO |
|---|---|---|
| 生成对象 | 一段回答 | 多轮 action 与环境 observation 交替 |
| 权重粒度 | 原本就可以逐 token | 仍可逐 token，也可共享 step 权重或联合选择 |
| 常见收益代理 | 同一 response 优势乘各 token 的方向 alignment | 简单基线仍可用轨迹优势；进一步研究真实历史下的条件后续收益 |
| 移动代价 | 局部 Fisher 二次代价 | 同样可用局部 Fisher；完整动作 KL 的有限移动可能耦合不同位置 |
| 新的难点 | 方向是否值得跟随 | 还要处理环境状态、延迟结果、历史对齐和事后信息依赖 |

“原版是序列级，Agent 才首次推导 token 权重”并不准确。序列级通常描述奖励来源或生成单元，不意味着所有 token 的权重相同。

设某个位置的收益为 $b$、正代价为 $c$、惩罚强度为 $\tau>0$，候选二次目标是

$$
\max_{0\le w\le\bar w}\;bw-\frac{\tau}{2}cw^2.
$$

先对 $w$ 求导，得到 $b-\tau cw$；内部驻点为 $b/(\tau c)$，再考虑区间约束：

$$
w^*=\operatorname{clip}\left(\frac{b}{\tau c},0,\bar w\right),\qquad c>0.
$$

这回答“给定收益和代价后，强度怎么选”，没有回答“收益估计是否可信”。若 $c=0$，应单独处理退化的线性目标，不能直接除以零。原实现的平滑式 $h/(h+\alpha_c c+\epsilon_c)$ 是另一种控制设计；二者不能当成同一个公式。

联合版本先写 $\mathbf w^\top\mathbf B-\tau K(\mathbf w)$。只有进一步采用对角二次近似，才得到可逐位置求解的比值。若加入整步启用成本，还要比较最优开启效用与关闭效用；详细推导、约束与零代价处理见完整稿第 12–15 节。

## 3. 为什么可以不用 FEC

FEC 从正、负、无证据 Teacher 分布构造任务方向和干扰方向，再做 Fisher 几何下的投影。它试图回答“Teacher 的改变里，哪些是任务相关纠正，哪些是共同偏移”。

Agent 中有另一类直接证据：工具执行错误、API 返回、检索结果、状态变化、测试结果以及任务终局。可以先从这些证据构造方向，不要求先收集正确／错误兄弟证据，更不要求保留 FEC 投影。

但“真实 observation”不意味着它引起的所有概率变化都有益。加长上下文、修改回放格式、看到后面的动作，都可能改变分数；执行成功也不必然等于推进任务。因此，去掉 FEC 可以作为独立设计，不能据此宣称所有干扰都已消失。

## 4. 新检索文献：哪些最接近这条路线

本次重新检索了更新后的 Obsidian `Agent OPD 专题`，特别是新增／更新的 TurnSight 笔记，并核对下列直接相关论文的公开原文。表内是各自论文的方法摘要，不是本仓库实验结果。公开链接用于保证 GitHub 文档可独立阅读。

| 工作 | 方法要点 | 与本讨论的关系 |
|---|---|---|
| [TurnSight](https://arxiv.org/html/2608.04007) | 真实工具执行结果作为 hindsight；比较 1、2、3 轮视界，方向投票后形成回合级信号，调节 RL 优势幅度 | 最直接支持用自身工具返回构造监督；其分析中仅工具结果优于额外加入标准答案 |
| [OCSD](https://arxiv.org/html/2608.04788) | Full 与 Observation-Ablated 使用匹配的回放格式；相减得到观测残差，在高 NLL 步调节 GRPO 幅度 | 为“去掉 FEC 投影后怎样保留针对性对照”提供直接基线 |
| [AHEAD](https://arxiv.org/html/2608.24114) | 分析失败轨迹并定位错误；常规步用环境反馈，错误步补充纠错提示；成功轨迹绕过该流程 | 将全局理解转为局部指导；通过有界 advantage 重加权接入 RL |
| [ICSD](https://arxiv.org/html/2608.14945) | 用教师方向对局部 RL surrogate 的一阶影响校准蒸馏权重，动作轮内保持辅助权重总量 | 与 VERPO 的目标对齐问题直接相关；不能把教师支持度等同于优化收益 |
| [HERO](https://arxiv.org/html/2606.11559) | 复盘全轨迹并生成逐轮反馈，再以反馈增强的自教师进行 action-token 蒸馏 | 支持“全局复盘 → 局部 Teacher 指导”；其反馈元数据不应被误写为已实现的数值置信 gate |
| [SEED](https://arxiv.org/html/2607.14777) | 从完整轨迹提炼 hindsight skill，比较技能与普通上下文下的原轨迹 token 支持，结合门控自蒸馏与 RL | 是全局经验压缩的另一种实现，区别于直接使用原始 observation |
| [SDPO](https://arxiv.org/html/2601.20802) | 把反馈作为额外上下文，重评原先采样的 token，再进行自蒸馏 | 说明额外反馈与 teacher-forcing 可以结合，不需要额外生成一条 Teacher 答案 |
| [AR-OPD](https://arxiv.org/html/2606.10385) | 部分特权视图作为 anchor，完整特权视图作为受控的 log-space residual | 提醒区分事后正确与当前前缀可学；不是 Agent 环境信用已经解决的证据 |

更新后的专题还包含 BCSD、UCOB、OVCSD、CRAFT、StepOPSD 等笔记，可作为后续查新入口。尤其不能因为标题里出现 bidirectional，就把它理解为 BERT 双向注意力：BCSD 讨论的是互补技能上下文，UCOB 讨论的是跨视图蒸馏方向。这些名称不构成更改注意力掩码的依据。

综上，“未来 observation + Teacher replay”已经有直接先行工作；本方案不能把这一步本身作为新颖性主张。剩余研究重点是证据选择、可验证的纠错、收益校准及计算成本。

## 5. Notation：本方案的最小符号表

| 符号 | 含义 |
|---|---|
| $\mathcal T$ | 一条完整交互轨迹，避免与前文 KL 系数 $\tau$ 混淆 |
| $k,t,v$ | 第 $k$ 个交互 step、该步第 $t$ 个策略 token、候选词表 token |
| $h_k$ | 生成第 $k$ 步前已知的任务、历史与当前 observation |
| $a_k$ | 第 $k$ 步策略生成片段，包含纳入训练的推理和可执行动作 |
| $o_{k+1}$ | 执行 $a_k$ 后返回的 observation；相对 $a_k$ 属于未来信息 |
| $E(\mathcal T)$ | 从完整轨迹提取的全局反馈前缀 |
| $E^{\mathrm{abl}}$ | 移除目标反馈内容、保留匹配格式的对照前缀 |
| $p_{k,t}$ | Student 在原始因果上下文中的词表分布 |
| $q^{\mathrm{obs}}_{k,t},q^{\mathrm{abl}}_{k,t}$ | 同一 Teacher 快照在有反馈／消融反馈前缀下的分布 |
| $q^0_{k,t}$ | 独立的原始上下文 Teacher 分布；不与消融回放分布混为一谈 |
| $e_{k,t}$ | 已采样 token 的观测 log-probability 残差 |
| $m_{k,t}$ | 策略生成位置掩码；环境、特权前缀、padding 等为零 |
| $A_i$ | 第 $i$ 条轨迹的 RL 优势，不能自动视为准确的局部 action 优势 |

## 6. 全局信息怎样进入自回归 Teacher

### 6.1 直接按时间顺序回放为什么不够

```text
任务 → action₁ → observation₁ → action₂ → observation₂ → 最终结果
```

整条字符串一次送入普通 causal Transformer，计算 action₁ 的 logits 时仍看不到右侧 observation。Teacher-forcing 可以并行计算各位置，但不会解除 causal mask。

### 6.2 当前确认的版本：全局反馈前置，整条轨迹回放

```text
Teacher 输入：
  [训练用特权反馈前缀]
  带 step ID 的工具返回、observation、最终结果
  或从完整轨迹提炼的复盘反馈

  [原轨迹回放区]
  任务 → action₁ → observation₁ → action₂ → observation₂ → action₃
         ↑取分布                   ↑取分布                   ↑取分布

Student 输入：
  任务 → action₁ → observation₁ → action₂ → observation₂ → action₃
         ↑计算损失                 ↑计算损失                 ↑计算损失
```

交互时间上的未来，被放到了回放文本的前缀。Teacher 在每个 action 的预测位置都能利用这个前缀，仍然保持自回归结构。

形式上，同一位置比较的是

$$
p_{k,t}(v)=\pi_\theta(v\mid h_k,a_{k,<t}),\qquad
q^{\mathrm{obs}}_{k,t}(v)=\pi_{\bar\theta}(v\mid E(\mathcal T),h_k,a_{k,<t}).
$$

这里是对额外上下文的条件化，而不是把 $q^{\mathrm{obs}}$ 解释成不依赖未来的在线行为策略。

**一次完整的 Teacher 分支前向即可覆盖所有有效 step 的 action 位置**。如果另有消融分支，就还需要一次该分支的回放；两分支可按资源批处理。超长轨迹可能需要分块、重算或缓存，不能把“一次逻辑回放”承诺成任何长度下的一次物理 GPU 调用。

Student／Teacher 的前缀长度不同，必须保存原轨迹位置到各分支预测 logits 的显式映射，不能使用同一个绝对下标直接取值。反馈、environment observation、padding 不参与策略 loss；原始 observation 继续作为后续 action 的输入。

### 6.3 什么时候才需要逐 step 回放

若所有 step 共享同一个 $E(\mathcal T)$，可以整轨迹回放。如果要求“评 action₁ 只给反馈₁，评 action₂ 只给反馈₂”，则通常需要独立的 step 样本或专门的可见性掩码；独立样本也可以 batch 处理。

把所有局部反馈写在同一个全局前缀，只是按 step ID 组织文本，不会强制每个 action 只能看自己的反馈。这个版本应称为全局共享证据，而不是严格的局部可见性控制。

## 7. 全局评价、token 概率和动作价值不是一个东西

全局分析器可以读取完整轨迹，然后输出对某个 action 的诊断。它的输出位置在轨迹之后，因此普通自回归模型就能看全局。它可以和 replay Teacher 共用模型参数，但角色不同：前者产出反馈，后者产出对原 action token 的条件分布。

Teacher 看到结果后提高某 token 的概率，表示该 token 在新上下文下更受支持，不等于这个动作提高了预期回报。观察到执行结果也不等于观察到了替代动作的结果。

例如原轨迹先查询用户 ID，收到 ID 后再查订单。全局 Teacher 知道 ID，不能据此要求早期 Student 跳过必要查询、直接使用尚未知晓的 ID。可学的反馈应是“先获取 ID，再调用需要它的接口”，而非复述未来专属数值。

因此全局前缀的首个版本优先放带来源的 observation／反馈，不直接复制全部待评分 action 正文或标准答案。即使不显式复制 action，observation 也可能包含动作回显，仍需检查复制捷径。复盘生成器的提示、证据 ID、原始反馈与最终反馈都应可追溯。

## 8. Observation 残差如何替代 FEC 方向构造

最直接的校准量来自匹配的两种 Teacher 条件：

$$
e_{k,t}=\log q^{\mathrm{obs}}_{k,t}(a_{k,t})
-\log q^{\mathrm{abl}}_{k,t}(a_{k,t}).
$$

它问的是“真实反馈相对相同格式的空反馈，使这个 token 的支持改变了多少”。OCSD 使用这个 log-probability 对照，并通过有界信号调节 GRPO；它没有实现本仓库 FEC 投影。[OCSD 原文](https://arxiv.org/html/2608.04788)

我们的整轨迹版本是受该对照思想启发的候选适配，不等于逐项复现 OCSD。全局前缀中若还含复盘结论，仅删除原始 observation、却保留由它推导的答案或诊断，便不能称为“移除全部观测信息”。原始 observation 和生成式复盘应分别做消融；格式匹配也不保证语义、长度和所有交互效应完全抵消。

如果保留 VERPO 的独立 evidence loss，可以使用完整词表方向：

$$
\Delta^{\mathrm{obs}}_{k,t}(v)=q^{\mathrm{obs}}_{k,t}(v)-q^{\mathrm{abl}}_{k,t}(v),
\qquad
\ell^{\mathrm{FKL}}_{\mathrm{evi},k,t}
=-w_{k,t}\sum_v\Delta^{\mathrm{obs}}_{k,t}(v)\log p_{k,t}(v).
$$

这是候选 signed correction loss，不是 OCSD 的原目标。$q^{\mathrm{obs}}$、$q^{\mathrm{abl}}$ 和 $w$ 停止梯度时，独立 Student logits 上的下降方向为 $w\Delta^{\mathrm{obs}}$。消融分布仍含回放格式，不应替换独立 reference 分支 $q^0$ 而不说明目标变化。

RKL 对应采用 log-density 方向 $r=\log q^{\mathrm{obs}}-\log q^{\mathrm{abl}}$ 和损失 $-w\sum_v p(v)r(v)$；其下降方向是 $wF(p)r$。不能把 FKL 概率差直接当成 RKL 的同一个梯度方向。

## 9. 三条训练路线与建议的验证顺序

| 路线 | 训练方式 | 要回答的问题 |
|---|---|---|
| 直接反馈 OPD | RL + 对有反馈 Teacher 的独立蒸馏项 | 额外反馈本身是否有用？ |
| 反馈 advantage 重加权 | 用有界反馈信号调节原 RL 优势，参考 OCSD／TurnSight | 保留原奖励符号时，局部幅度分配是否改善？ |
| 观测残差 VERPO | RL + 独立 reference + 有／无观测 Teacher 残差纠正 | 增量方向是否优于绝对模仿，是否还需要自适应权重？ |

三者不是同一个目标。特别是 $\widetilde A=A\cdot w$ 在 $A=0$ 时仍为零；全错同奖励组不会因此自动获得额外更新。独立蒸馏分支可以有非零梯度，但若再次用原 $A\zeta$ 的正收益 gate 将它完全关掉，仍会失去这条学习通道。

建议先用固定辅助权重、相同 reference 强度和相同 loss reduction 比较上述证据／目标选择，再单独增加 ZPD 控制器。固定权重不是最优性的声明，而是避免同时改变证据和控制器后无法归因。

第一阶段使用共享的原始全局 observation 前缀；第二阶段增加 matched ablation；第三阶段单独比较全局复盘与相关 step 反馈；最后再检验收益估计、token／turn 权重和额外控制。每阶段记录 Teacher token 开销与环境交互预算，不能仅比较 optimizer step 数。

## 10. 整轨迹回放训练伪代码

下面明确选择“观测残差 FKL + 固定辅助权重”作为研究示例。它没有 FEC，也没有逐 step Teacher 调用。Teacher 调度、系数和 reduction 都由未来明确注册的配置决定；伪代码不改变现有训练设置。

```python
# 研究伪代码，尚未接入生产训练器。
for iteration in training_iterations:
    # Student 在环境中正常交互，只使用当时可见的历史。
    groups = rollout_student_in_environment()
    save_actions_observations_rewards_old_logprobs(groups)
    advantages = compute_registered_rl_advantages(groups)

    for trajectory in groups:
        # 当前起点：全局原始反馈，不启动额外 LLM 复盘。
        evidence = collect_global_observations_with_step_ids(trajectory)
        ablated = remove_feedback_content_keep_matched_fields(evidence)

        student_input = original_causal_trajectory(trajectory)
        full_input = prepend_feedback(evidence, student_input)
        ablated_input = prepend_feedback(ablated, student_input)
        maps = align_original_action_predictor_positions(
            student_input, full_input, ablated_input
        )
        store_replay_inputs_and_policy_mask(trajectory, maps)

    # 一次 minibatch 内冻结 Teacher 分支；不是对每个 step 再生成动作。
    for minibatch in policy_minibatches(groups):
        with no_grad():
            q_obs = teacher_replay_full_trajectories(minibatch.full_inputs)
            q_abl = teacher_replay_full_trajectories(minibatch.ablated_inputs)
            q_ref = teacher_replay_full_trajectories(minibatch.student_inputs)
            q_obs, q_abl, q_ref = gather_aligned_policy_distributions(
                q_obs, q_abl, q_ref, minibatch.maps
            )
            delta = q_obs - q_abl
            weights = fixed_weight_on_policy_positions(minibatch)

        p = student_replay_full_trajectories(minibatch.student_inputs)
        p = gather_aligned_policy_distributions(p, minibatch.student_map)

        # PPO ratio 使用保存的 rollout old_logprob，不使用 Teacher logprob。
        loss_rl = registered_rl_loss(p, minibatch.old_logprobs, advantages)
        loss_ref = registered_reduce(kl(q_ref, p), minibatch.policy_mask)
        loss_evi = registered_reduce(
            -weights * sum_over_vocab(delta.detach() * log(p)),
            minibatch.policy_mask,
        )
        loss = loss_rl + lambda_ref * loss_ref + lambda_evi * loss_evi
        success = optimizer_step(loss)
        if success:
            update_teacher_according_to_registered_schedule()

    evaluate_student_without_privileged_prefix()
```

这里 $q_{\mathrm{ref}}$ 指配置选定的 Teacher 在普通上下文下的分布。若实验使用独立冻结 reference 模型，必须显式替换该分支并记录，不能悄悄改变 reference 的身份。

伪代码展示 full-vocabulary 数学语义，不要求保存整个 token × vocabulary 张量。实际实现可分块计算；若采用仓库现有 top-k 截断合同，应记录丢弃质量和近似，不能通过悄悄重归一化改变公式含义。轨迹长于上下文上限时须另定截断／分块方案。

## 11. 理论边界与必要核对

原 Agent 可预测方向分析要求：方向在当前前缀处预先确定，不能依赖同一目标动作之后产生的随机结果。当前 $E(\mathcal T)$ 明确依赖这些结果，因此整轨迹 hindsight 方案应归入**事后监督训练**。它仍可计算并优化，却不能直接继承完整稿中条件 score 零均值、同历史收益无偏和整体回报改进的保证。

“停止梯度”仅阻止计算图反传，不会消除统计依赖；“用独立轨迹训练预测器”也不自动消除 Teacher evidence 对当前目标未来的依赖。要建立在线控制保证，需要另行定义只依赖可见前缀的控制场及独立估计过程。

实现与小规模试验需要核对：

- 普通 Student 回放与采样前缀语义一致；Teacher 前缀变化后，原 token／logit 对齐仍准确。
- Teacher 的分布和控制权重 detach；策略 loss 不落在工具 observation 或反馈前缀上。
- 普通因果分支中，改写未来不会影响早期 logits；hindsight 分支中，改写反馈前缀可以影响早期 logits。后者在此方案中是预期行为。
- 有观测／消融观测分支使用同一 Teacher 快照和匹配格式，保留原始反馈、消融规则和生成式反馈的来源。
- 分开统计失败 action、合法但无助的 action、有效信息查询与最终成功；执行合法性不替代任务收益。
- 审计未来 ID／答案复制、工具跳步、提示回显、格式模仿；保存改进和退化的真实轨迹。
- 评测只使用普通 Student 提示；记录成功率、工具成本、长度、错误恢复，以及 Teacher replay 的实际 token／时间开销。

## 12. 已完成与未完成

已完成：串联原 VERPO 和 Agent 的推导，补充符号表与训练伪代码，修复原报告公式分隔符及表格内公式；检索更新后的 Agent OPD 文献，并明确整轨迹 hindsight 回放的候选设计。

尚未完成：真实 Agent 数据管线、全局反馈打包、分支 token 对齐实现、训练协议注册、GPU 实验和收益验证。文中伪代码、推荐消融与论文报告结果均不代表本仓库已经运行这些实验。

完整数学与原基线伪代码：[VERPO 到 Agent VERPO 完整推导](VERPO%E5%88%B0Agent_VERPO_%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md)。原联合优化参考内核及数值核对入口：[目录 README](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/agent%20opsd/agent_verpo_report/README.md)。

## 13. 后续澄清与独立仓库决策

当前方案明确不需要 step gate，也不增加收益预测网络。策略 token mask 仅排除 observation、padding 和其他非策略目标，不是 step gate。是否使用已有 token 权重公式与是否训练预测网络是两个问题；固定权重是第一轮归因对照，原 token 控制器作为独立消融。

此前提到的收益预测器预测的是“沿指定 Teacher 方向调整预计带来的收益”，区别于常规 value 网络预测的剩余回报。增加这种网络会更接近 Actor–Critic 的辅助估计结构，但不自动等于 PPO；PPO 的裁剪更新目标与是否使用独立 critic 是不同设计维度。

若以后研究轻量版本，可复用 Student 最后一层、停止梯度的 hidden，接 `4096→256→1` 的 MLP。以 Qwen3-8B 为例，该头约 105 万参数，常规 Adam 的参数／梯度／状态约 16–18 MiB；一份 BF16 hidden 缓存按 `本卡驻留 token 数 × 4096 × 2 字节` 计算，8K／32K token 分别约 64／256 MiB。每卡 8K–32K 驻留 token 下可先预留 0.1–1 GiB 额外空间，属于实现相关预算，不是实测峰值。完整约 8B critic 的全参数训练状态按每参数 16–18 字节估算合计约 119–134 GiB；理想 8 卡全分片后约 15–17 GiB/卡，另加激活、通信和临时缓冲。依据：[Qwen 配置](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)、[训练显存组成](https://huggingface.co/docs/transformers/v4.50.0/model_memory_anatomy#anatomy-of-models-memory)。

收益预测器可能降低单条轨迹信号噪声、共享类似状态经验，但会增加标签偏差、冷启动、非平稳性和校准成本。拟合 $A\zeta$ 不会自动获得真实因果收益，也不会默认节省 Teacher replay。没有实验依据预估成功率提升百分点，因此当前先不用。

新实验在 [VERPO-AGENT](https://github.com/hamsterjiang23/VERPO-AGENT) 独立维护，通过固定提交的 Git submodule 引用本仓库的 veRL 与 VERPO 公共代码。新仓库负责环境、回放适配、独立配置和评测；共享修复回到本仓库。现有 SDPO／RLCSD launcher 的历史训练配置不作为新 Agent 实验的默认协议。新仓库初始交付是可检查的项目骨架与依赖接入，不是已完成的 Agent 训练器。
