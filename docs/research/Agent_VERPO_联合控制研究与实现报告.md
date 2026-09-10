# Agent VERPO：基于终局回报与因果前缀回放的 Step–Token 联合纠正控制

> 上游研究快照，来源提交 `9c01f2bcd05aea1b3741ba60ede9aad201e0875c`。历史候选中的 FEC、step gate 或收益预测器不覆盖[当前设计](../current_design.md)。原文与导出文件指纹见 [provenance.json](provenance.json)。

> **首次阅读建议：** 先看[从 VERPO-ZPD 到 Agent VERPO 的完整推导](VERPO%E5%88%B0Agent_VERPO_%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md)。该稿从原有蒸馏分解、证据方向和 ZPD 开始，补齐本报告之前的推导链。

**版本：v2 · 日期：2026-09-10**\
**性质：数学推导、算法设计与有限离散核对；不是已完成的 Agent 训练实验。**

## 摘要

目标是在多步 Agent 中保留 VERPO 的教师证据方向与 Fisher 几何，利用旧 Student 产生的真实交互轨迹和终局回报，联合决定：**哪些 step 接受纠正、这些 step 中哪些 token 接受纠正、各接受多少。**

本报告保留原报告的方向收益定义，不把 Teacher 置信度当作环境收益，不把失败轨迹的每个 token 当作错误，也不要求 Teacher 重新生成整条替代轨迹。基础数据仍是：每个 prompt 的 8 条 Student rollout，以及 Teacher 与旧 Student 在真实因果前缀上的前向回放。

主要结论如下。

1. 在冻结策略、方向可预测、采样分布一致等条件下，终局回报乘以中心化 token 分数可以估计局部纠正方向的价值导数；token 条件 Fisher 可以由完整词表直接计算。
2. 8 条轨迹共享初始历史，但通常不共享后续历史。它们提供大量局部训练样本，不自动提供每个中间前缀的 8 样本条件估计。
3. 先按原始整步收益拒绝 step，再挑 token，会漏掉“整体方向不好、局部子集有益”的情况。更合理的是先求解 token/方向子集，再判断其净收益是否值得激活该 step。
4. 在固定、可预测的方向基底上，可以建立精确的联合线性动作分布优化；在零点 Fisher 对角近似下，可以得到低成本的 token 权重闭式解和 step 激活规则。
5. 必须区分**事后训练权重**与**生成前可执行的控制策略**。用本条轨迹的最终回报给本条 token 加权可以实现训练，但不自动满足非预知策略的改进定理。
6. 推荐先实现可诊断的 replay-only 版本，再用独立/交叉拟合的前缀收益预测器降低同样本选择偏差；理论核对使用少量可回放状态和独立局部干预，不把额外分支采样强加给全部训练数据。

### 来源与证据等级

- **[原稿]**：用户上传的 `Pasted markdown(20260910-064456).md`，标题为《Agent OPSD 中 VERPO 的 step 与 token 联合控制：详细推导》。本报告引用其可见行号与章节；其中“源码已验证”的描述属于原稿的记录，本次没有重新访问或核对该仓库。
- **[本文推导]**：在明确假设下，从原稿公式进一步推出的数学结果。有限脚本核对不是完整形式化证明。
- **[工程提案]**：为接入训练提出的实现、诊断和实验设计，尚无真实模型训练结果。
- **[外部基础]**：RLOO 原论文与 TRPO 原论文只用于确认相应基础机制，不用于宣称本方案新颖或已证明实测有效。

---

## 1. 先明确最终要实现的对象

### 1.1 三个决策不是三个互不相关的分数

令第 $k$ 个 step 内，token 前缀 $u_{k,t}$ 的纠正权重为 $w(u_{k,t})\geq0$。

- **在哪里纠正**：权重为正的策略 token 位置。
- **纠正多少**：各位置的权重，以及实际训练后测得的分布变化。
- **哪个 step 被纠正**：该 step 内至少有一个位置被接受；如需在生成整步之前预筛选，则另需只依赖 $h_k$ 的 step 控制器。

这里的“纠正”是沿既定教师方向改变分布，不是认定原 token 错误，也不是把生成文本直接替换成教师文本。[原稿 §4–5，L104–174]

### 1.2 两种运行语义必须分开

| 版本 | 权重使用的信息 | 实际作用 | 能声称什么 |
|---|---|---|---|
| **R0：事后 replay 加权** | 当前已完成轨迹的回报、采样 token、前缀方向 | 构造一次 Student 训练更新 | 可实现的经验训练算法；不能直接声称该 gate 是合法的在线策略或无偏选择 |
| **R1：前缀条件控制** | 当前 token 产生前可获得的前缀特征、冻结预测器和固定方向 | 定义可预测的纠正场，再蒸馏或用于受控干预 | 可接入条件收益与 Fisher 推导；仍需处理预测误差、可行性和参数投影 |
| **T：有限状态严格核对** | 同历史的充分独立样本，或可枚举精确价值 | 求精确联合目标、检查 KKT 与局部改进 | 仅在相应有限问题和条件下成立的数学/数值结论 |

**建议实施顺序：R0 建管线与反例诊断 → R1 降低选择偏差 → 用 T 校准局部机制。**不要把 T 的定理直接写成 R0 的性能保证。

---

## 2. Agent 建模与数据约定

### 2.1 交互过程

为简化主线，先取有限时域、$\gamma=1$、终局奖励 $R\in[0,1]$。实际接入保留任务原有 reward 尺度，不要求把 ±1 奖励改成 0/1；任意有界区间 $[r_{min},r_{max}]$ 的矩约束将后文的 1/4 替换为 $(r_{max}-r_{min})^2/4$，相关罚项也须使用同一收益单位。扩展到稠密奖励时使用剩余回报 $G_k=\sum_{j\ge k}r_j$，不是在所有 step 上无条件复制总回报。

$$
h_k=(x,a_1,o_1,\ldots,a_{k-1},o_{k-1}),\qquad
 a_k=(y_{k,1},\ldots,y_{k,T_k}). \tag{1}
$$

$$
u_{k,t}=(h_k,y_{k,<t}),\qquad
p_k(a_k\mid h_k)=\prod_{t=1}^{T_k}p(y_{k,t}\mid u_{k,t}). \tag{2}
$$

$p$ 是本次 rollout 实际使用的冻结旧 Student；它不是更新中的 $\pi_\theta$，也不是无证据 Teacher anchor $q^0$。[原稿 §3、§8，L52–80、L252–256]

### 2.2 策略 token 与 observation

必须保留全部真实 observation 作为后续上下文，但 observation 不进入策略概率的乘积、纠正收益求和或策略输出损失。包括 reasoning、工具名、参数、最终回答、EOS 在内的所有实际策略生成内容均需有明确 mask。

```text
prompt
  → policy tokens of step 1        [计入 score / Fisher / 训练目标]
  → tool observation 1             [保留上下文，不作为策略输出监督]
  → policy tokens of step 2        [计入]
  → tool observation 2             [同上]
  → final answer / EOS             [计入]
```

Mask observation 的输出目标，不等于要求后续损失对 observation 的输入表征停止梯度。二者不是同一操作。[本文工程澄清]

无效工具调用、错误 JSON、可生成但导致失败的动作必须保留为失败轨迹；不能事后删除这些样本来改善收益统计。硬约束采样使用的合法支持则必须在分析分布中保持一致。[原稿 L58–64]

### 2.3 冻结与可预测性假设

在一个分析/更新批次内固定：旧 Student、Teacher/EMA Teacher 快照、方向构造规则、采样设置、环境版本与奖励规则。每个方向 $s(u,v)$ 必须在抽取当前 token $y$ 之前由前缀与已固定证据确定。

**合法例子**：训练集事先提供的参考解答；生成当前 token 前已出现的工具结果；独立数据固定的正负证据池。

**不能直接套用无偏定理的例子**：看完目标轨迹才挑选的正确 sibling；当前轨迹未来的 observation；先依据本条成功/失败路由教师，再用本条奖励评估同一方向。

冻结 Teacher 参数不消除这些数据依赖。独立证据仍可能含训练专用信息：这不自动等于未来泄漏，但部署时不可见的信息还需要边际化/投影分析。[原稿 L78–83、L350]

### 2.4 必须记录真正的行为分布

如果 rollout 按 $p_b=\operatorname{softmax}(z/T+\text{mask})$ 采样，回放就必须用同一个 $p_b$。使用温度 1 的原始分布去评估温度 0.7 采样的轨迹，会破坏后文的条件零均值。若 $s$ 定义在原始 logits 坐标，温度固定时分析采样 logits 应使用 $s/T$。这是对式 (2) 求导的直接结果。

动态 top-p 截断可能使支持随扰动改变。首版理论核对建议关闭动态截断，或明确把固定截断支持作为被分析的行为策略。不要在采样后把截断概率重归一化，再声称仍估计原全词表策略。

---

## 3. Teacher 回放到底提供什么

### 3.1 输入输出契约

在每个有效策略前缀 $u$ 上，需要：

$$
p(v\mid u),\qquad s(u,v),\qquad y,\qquad \text{step id},\qquad G_k. \tag{3}
$$

Teacher 提供的是词表方向，不是“这个 token 得了 80 分”。同 tokenizer 的 FKL 证据残差可作为原稿约定的 logit 下降方向；FEC 则使用完成 nuisance 去除后的方向。RKL 也要明确输出到底是原始比值、概率位移，还是已经映射后的 logit 下降方向。[原稿 L39–46]

**统一接口建议**：

```python
s = build_frozen_logit_direction(
    behavior_probs=p,
    teacher_distributions=teacher_prefix_replay,
    evidence_config=frozen_config,
)  # [vocab]，尚未乘本条回报导出的 gate/advantage
```

本文不改写原 VERPO 的 reference restoration 或 FEC 公式；新控制器接在“原始可评估方向”之后。若改变了 clipping、词表截断、nuisance 系数或已有 gate，必须用最终被评估方向重算全部统计量。

### 3.2 因果回放不等于逐 token 单独调用模型

固定完整历史后，可通过符合模型 attention 语义的批量 teacher forcing 取得各预测位置 logits。序列后缀不会经标准因果 attention 影响较早位置，但若把未来 observation 放入开头的 evidence/system prompt，仍然会泄漏。

预测 $y_t$ 的特征必须来自 $u_t$，不能用已经读到 $y_t$ 的隐藏状态充当前缀特征。Teacher 与 Student 的 evidence 前缀长度不同，response 对齐必须按显式索引映射，不可直接拿同一个绝对 token 位置。

---

## 4. 从 token 方向推导收益与代价

以下推导以固定前缀 $u$ 为条件；Teacher 方向已固定。均沿用原稿的 logit 坐标。

### 4.1 中心化 score

定义微扰：

$$
p^\epsilon(v\mid u)=
\frac{p(v\mid u)e^{\epsilon s(u,v)}}{\sum_b p(b\mid u)e^{\epsilon s(u,b)}}. \tag{4}
$$

对对数求导：

$$
\left.\partial_\epsilon\log p^\epsilon(y\mid u)\right|_0
=s(u,y)-\sum_vp(v\mid u)s(u,v)
\equiv\zeta(u,y). \tag{5}
$$

因此：

$$
\mathbb E_p[\zeta\mid u]=0. \tag{6}
$$

$\zeta$ 是“沿当前方向移动时采样 token 的对数概率导数”，不是正确率，也不是 Teacher 置信度。[原稿 §4]

### 4.2 token 条件方向收益

令当前 token 之后，本 step 剩余生成、环境执行及后续交互都由旧 Student 继续。定义：

$$
Q^p(u,v)=\mathbb E_p[G_k\mid u,y=v]. \tag{7}
$$

只在这个前缀沿式 (4) 扰动，则价值的一阶导数为：

$$
\begin{aligned}
b(u)
&=\left.\partial_\epsilon\sum_vp^\epsilon(v\mid u)Q^p(u,v)\right|_0\\
&=\sum_vp(v\mid u)Q^p(u,v)\zeta(u,v)\\
&=\mathbb E_p[(G_k-\beta(u))\zeta(u,y)\mid u].
\end{aligned}\tag{8}
$$

只要 $\beta(u)$ 在当前 token 采样前确定，并满足相应独立性，就不改变期望。

- $b(u)>0$：接受这份方向存在正的一阶局部收益。
- $b(u)<0$：该方向单独启用时一阶不利。
- 单个 $(G_k-\beta)\zeta$ 的正负，不等于 $b(u)$ 的正负。

式 (8) 是原稿 token 价值分解的条件形式，不是新增的逐 token 正误标注器。[原稿 L150–174、L378–382]

### 4.3 token Fisher 代价

$$
\begin{aligned}
c(u)&=\mathbb E_p[\zeta(u,y)^2\mid u]\\
&=\sum_vp(v\mid u)\left(s(u,v)-\mu(u)\right)^2\\
&=s(u)^\top F(p_u)s(u),\\
F(p)&=\operatorname{diag}(p)-pp^\top,\qquad
\mu(u)=\sum_vp(v\mid u)s(u,v).
\end{aligned}\tag{9}
$$

不需要创建 $V\times V$ 的 Fisher 矩阵。给定已访问前缀，完整词表上这个数可以直接计算；未访问前缀的分布仍需采样或预测。

### 4.4 从 token 累积到 step

对固定 $h_k$，令：

$$
S_k(a)=\sum_{t=1}^{T_k}\zeta(u_{k,t},y_{k,t}),\qquad
 d_k(a)=p_k(a\mid h_k)S_k(a). \tag{10}
$$

由条件零均值得 $\sum_a d_k(a)=0$。

$$
B_k(h_k)=\mathbb E_p[G_kS_k\mid h_k]
=\mathbb E_p\left[\sum_t b(u_{k,t})\mid h_k\right]. \tag{11}
$$

对于 $t<r$，$\zeta_t$ 已是第 $r$ 个 token 前可测量的量，因此：

$$
\mathbb E[\zeta_t\zeta_r\mid h_k]
=\mathbb E[\zeta_t\mathbb E[\zeta_r\mid u_r]\mid h_k]=0. \tag{12}
$$

于是：

$$
C_k(h_k)=\mathbb E_p[S_k^2\mid h_k]
=\mathbb E_p\left[\sum_t c(u_{k,t})\mid h_k\right]. \tag{13}
$$

这是期望恒等式，不是单条轨迹上 $S_k^2=\sum_tc_t$。完整 step 目标对应 token 求和；逐条除以长度会改变目标。[原稿 §5–6]

---

## 5. 从一个 prompt 的 8 条轨迹估计这些量

### 5.1 数据形状

可以把一次批次整理为：

```text
prompt_id, rollout_id, behavior_version, environment_version,
step_id, policy_token_index, prefix_hash,
sampled_token_id, behavior_logprob, policy_mask,
return_to_go, evidence_version, zeta, conditional_fisher
```

另外保存前缀特征、原始词表方向的可重算信息，以及所有 loss mask。完整词表张量可分块重算，不必全部长期落盘。

### 5.2 留一法中心化回报

同一 prompt 的 $M=8$ 条独立轨迹：

$$
A_i=R_i-\frac1{7}\sum_{j\ne i}R_j. \tag{14}
$$

对每个已访问 token 构造：

$$
X_{i,k,t}=A_i\zeta_{i,k,t},\qquad Y_{i,k,t}=c_{i,k,t}. \tag{15}
$$

$X$ 是条件方向收益的样本，$Y$ 是该前缀的 Fisher 值。这里只是原始回报中心化，不做组内标准差归一化。RLOO 的基础形式见 [外部基础 S1, §2.3]；将策略分数投影到固定教师方向，得到本文的标量版本。

若收益要对应环境终局成功率，就不应默默把 KL-shaped reward、GRPO 标准化 advantage 或本批次筛选权重当作同一个 $R$。它们可以构成别的 surrogate，但需要另行命名。

### 5.3 第一步具有 8 个共同历史样本

$$
\widehat B_1(x)=\frac18\sum_iA_iS_{i,1}
=\frac17\sum_i(R_i-\bar R)S_{i,1}. \tag{16}
$$

$$
\widehat C_1(x)=\frac18\sum_i\sum_{t\in step1}c_{i,1,t}. \tag{17}
$$

**后续 step 不可仅按 step 编号凑成同状态组。**观察、先前工具调用、reasoning、隐藏环境状态的条件分布都可能已经不同。[原稿 L372–376]

### 5.4 复用前面对话中的手算例子

假设每个 token 只有 A/B 两个选择，$p=(0.5,0.5)$，$s=(0.5,-0.5)$，每个 step 两个 token。每个 token 的 $c=0.25$。

| 轨迹 | 第一步 | 终局 $R_i$ | $S_{i,1}$ | $\sum_t c_{i,1,t}$ |
|---|---|---:|---:|---:|
| 1 | AA | 1 | 1 | 0.5 |
| 2 | AA | 1 | 1 | 0.5 |
| 3 | AB | 1 | 0 | 0.5 |
| 4 | BA | 1 | 0 | 0.5 |
| 5 | AB | 0 | 0 | 0.5 |
| 6 | BA | 0 | 0 | 0.5 |
| 7 | BB | 0 | -1 | 0.5 |
| 8 | BB | 0 | -1 | 0.5 |

$$
\widehat B_1=2/7\approx0.285714,\qquad\widehat C_1=0.5. \tag{18}
$$

这是人为构造的解释例子。本次代码实际核对了该结果。

### 5.5 中间前缀只有一个样本时的可实现方案

**R0：直接使用 $X_{i,k,t}$** 作为训练期经验收益，给已访问位置求权重。这无需额外预测网络，但必须把它标记为单样本经验 gate。

**R1：条件回归**。用只含当前前缀信息的特征 $f(u)$ 拟合：

$$
 b_\phi(f(u))\approx\mathbb E[X\mid f(u)]. \tag{19}
$$

若 $f(u)$ 没有保留所有相关信息，右边是同特征类的平均方向收益，不等于每个完整历史的精确 $b(u)$。这是必须记录的状态抽象误差。

完整词表 $c(u)$ 已可直接计算，通常不必再训练一个 cost head。只有打算在 Teacher replay 前预筛选、降低查询成本时，才需要提前预测代价或 step 价值。

为避免拟合本条噪声后又在本条上选 gate，按 **prompt/任务实例** 分组交叉拟合，整个 prompt 的 sibling 轨迹放在同一个 fold。方向构造数据也应与被评估 fold 保持需要的独立性。交叉拟合不等于获得有限样本符号保证，还要检查预测误差。

### 5.6 一个更适合规模化训练的解释：估计全局方向目标

对任意冻结、可预测的控制场 $\alpha(u)$，在所有策略 token 上施加微小 logit 方向 $\alpha(u)s(u)$，则：

$$
\left.\frac{d}{d\epsilon}J(p^{\epsilon,\alpha})\right|_0
=\mathbb E_p\left[\sum_{k,t}G_k\alpha(u_{k,t})\zeta_{k,t}\right]. \tag{20}
$$

这个结论由整条交互路径的 likelihood-ratio 求导得到；环境转移项不直接依赖策略参数。过去的奖励可由条件零均值消掉，从而使用 $G_k$。

**因此，不必先精确识别每个中间历史的 $B$，才能训练一个有意义的方向控制器。**不同历史上的 rollout 可以共同估计式 (20) 的全局旧策略分布目标；但不能把这个全局目标改称“所有具体前缀的真实局部收益均已被知道”。[本文推导]

---

## 6. 精确的 step–token 联合优化

### 6.1 先定义方向基底，而非直接把样本位置当作可部署策略

固定 $h_k$，选取 $d$ 个事先定义的可预测基底 $f_j(u)$。例如“本 step 第 $j$ 个生成位置”，或由当前前缀就能确定的 token 类别。不能用“最终失败 token”这种事后标签定义基底，再忽略其依赖。

$$
S_j(a)=\sum_t f_j(u_t)\zeta_t,\qquad
B_j=\mathbb E_p[G_k S_j\mid h_k]. \tag{21}
$$

定义联合线性动作路径：

$$
p_{\mathbf w}(a\mid h_k)
=p(a\mid h_k)\left(1+\sum_jw_jS_j(a)\right). \tag{22}
$$

要求所有完整合法动作上 $1+\mathbf w^\top\mathbf S(a)>0$。因为 $\mathbb E_pS_j=0$，式 (22) 归一化。

### 6.2 价值增益与 KL

旧 Student continuation 固定，因此式 (22) 的价值增益**精确**等于：

$$
\Delta U(\mathbf w)=\mathbf w^\top\mathbf B. \tag{23}
$$

$$
K(\mathbf w)
=\mathbb E_p[(1+\mathbf w^\top\mathbf S)\log(1+\mathbf w^\top\mathbf S)
-\mathbf w^\top\mathbf S]. \tag{24}
$$

最后的线性项期望为零；减去它能使逐样本 KL 形式非负，并避免有限样本分数均值漂移造成虚假的一阶 KL。[原稿 §11.5]

$$
\nabla K=\mathbb E_p[\mathbf S\log(1+\mathbf w^\top\mathbf S)],\quad
\nabla^2K=\mathbb E_p\left[\frac{\mathbf S\mathbf S^\top}{1+\mathbf w^\top\mathbf S}\right]\succeq0. \tag{25}
$$

所以：

$$
\max_{\mathbf w\in\mathcal W}\quad
\mathbf w^\top\mathbf B-\tau K(\mathbf w),\qquad\tau>0 \tag{26}
$$

是凸可行域上的凹最大化问题。只有在方向不退化、具有相应严格曲率时才能保证唯一解；线性相关方向下不可笼统声称唯一。

### 6.3 KKT 如何对应“选谁、多少”

先假设只需处理各坐标盒约束 $0\le w_j\le\bar w_j$，且整个盒已满足全支持可行性。令：

$$
g_j=B_j-\tau\partial_{w_j}K.
$$

最优点满足：

$$
\begin{cases}
g_j\le0,&w_j=0,\\
g_j=0,&0<w_j<\bar w_j,\\
g_j\ge0,&w_j=\bar w_j.
\end{cases}\tag{27}
$$

若还有耦合可行性约束，KKT 需要加入对应乘子。**不能独立求两个 gate 再相乘，便称为式 (26) 的联合最优。**[原稿 §13]

一般多方向问题中，即使某个 $B_j\le0$，也可能通过减少其他方向的联合代价而被选中；只有在后面的对角近似条件下，才得到简单的逐坐标正收益筛选。

### 6.4 为什么不同 token 在零点可以对角，但有限强度仍耦合

$$
C_{ij}=\mathbb E[S_iS_j]
=\mathbb E\left[\sum_tf_i(u_t)f_j(u_t)c(u_t)\right]. \tag{28}
$$

若每个基底作用于不重叠的 token 位置，$f_if_j=0$，于是零点 $C$ 对角。若多个教师方向作用于同一位置，或者方向是共享参数更新基底，交叉项不能省掉。

即使零点对角，式 (25) 的分母在非零 $\mathbf w$ 处也会重新引入耦合。本次有限例子中，零点跨 token 项约为 $10^{-18}$，非零强度下一个 Hessian 非对角项约为 **0.00481255**。这只是有限例子，不是实测模型结果。

---

## 7. 可落地的对角近似与 step 激活规则

### 7.1 局部二次目标

在零点附近：

$$
K(\mathbf w)=\frac12\mathbf w^\top C\mathbf w+O(\|\mathbf w\|^3). \tag{29}
$$

对角情况下，在收益尺度上加入可选的幅度惩罚 $\lambda_jw_j$ 和整步激活费用 $\kappa_s$：

$$
\max_{0\le w_j\le\bar w_j}
\sum_j\left[(B_j-\lambda_j)w_j-\frac\tau2 C_jw_j^2\right]
-\kappa_s\mathbf1\{\mathbf w\ne0\}. \tag{30}
$$

**这两个资源项是本文新增的设计，不是原稿自然推出的已有 VERPO 项。**$\lambda_j$ 是每单位强度的惩罚，不是严格 token 数；$\kappa_s$ 可以表示开一个 step 的固定代价，或仅作为稀疏正则。Teacher 已经全部 replay 完后，它不能被解释成节约了已付出的查询成本。

### 7.2 给定激活，token 强度的解

$$
\widetilde w_j
=\operatorname{clip}\left(\frac{B_j-\lambda_j}{\tau C_j},0,\bar w_j\right),
\qquad C_j>0. \tag{31}
$$

令：

$$
V_k^*=\sum_j\left[(B_j-\lambda_j)\widetilde w_j
-\frac\tau2 C_j\widetilde w_j^2\right]. \tag{32}
$$

由于“不开启任何 token”可取得零目标：

$$
\boxed{
\mathbf w_k^*=
\begin{cases}
\widetilde{\mathbf w}_k,&V_k^*>\kappa_s,\\
\mathbf0,&V_k^*\le\kappa_s.
\end{cases}}
\tag{33}
$$

这是对式 (30) 进行“关闭与最优开启”比较得出的结果，不是独立 gate 任意相乘。相等时约定不激活。

若无需稀疏资源约束，先令 $\lambda_j=\kappa_s=0$。原始 KL 正则并不自动产生 Top-K token 或固定 step 数。

### 7.3 为什么必须先看子集，再判断 step

构造可实现例子：两个独立 token score 各以 1/2 概率取 ±1，令回报 $R=0.5+0.2\zeta_1-0.3\zeta_2\in[0,1]$。因此 $B_1=0.2,B_2=-0.3,C_1=C_2=1,\tau=1$。

整份方向的收益为 $-0.1$，用一个共同强度的 step gate 会拒绝它。联合对角控制却得到：

$$
\widetilde{\mathbf w}=(0.2,0),\qquad V_k^*=0.02.
$$

若 $\kappa_s=0.01$，仍应接受第一个位置，净代理收益为 $0.01$。**原方向整体不利，不等于全部 token 子集不利。**本次代码核对了这一例子。

### 7.4 前缀条件版本与 step 预筛选

R1 用 $b_\phi(u)$ 代替条件收益、用可直接计算的 $c(u)$ 得到 $w(u)$。若不要求提前知道整个 step 是否启动，可以在各前缀依次决定；完成后统计该 step 是否有 token 被接受。

如需在 step 生成前就决定是否付出教师成本，需要预测：

$$
V^*(h_k)=\mathbb E_{a\sim p(\cdot\mid h_k)}\left[\sum_t
\left((b(u_t)-\lambda_t)\widetilde w(u_t)-\frac\tau2c(u_t)\widetilde w(u_t)^2\right)\right]. \tag{34}
$$

使用冻结的 $V_\psi(h_k)$ 预筛选。**不能用本 step 实际生成完后的 token 总效用，反过来声称那是生成前只依赖 $h_k$ 的 gate。**式 (34) 的新预测器仍需要独立估计和误差验证。[本文推导与工程提案]

### 7.5 数值稳定不是免费的理论步骤

实际经常写：

$$
w=\operatorname{clip}\left(\frac{\widehat b-\lambda}{\tau(c+\epsilon)},0,w_{max}\right). \tag{35}
$$

$\epsilon$ 相当于增加一个二次 ridge，改变了目标；原 VERPO 的 $h/[h+\tau(c+\rho)]$ 是另一种平滑 surrogate，不是式 (31) 的精确解。[原稿 L294–302]

若 $c=0$，方向在支持上等价于常数，理论上 $\zeta=0,b=0$。若预测器给出非零 $\widehat b$，应记录异常并保守回退，不应利用极小分母放大成强纠正。

### 7.6 可新增的结构约束：收益不能与代价任意组合

当条件剩余回报在 $[0,1]$ 时，由 Cauchy–Schwarz：

$$
|b(u)|^2=|\operatorname{Cov}(G,\zeta\mid u)|^2
\le\operatorname{Var}(G\mid u)c(u)\le\frac14c(u). \tag{35a}
$$

因此真实收益满足 $|b(u)|\le\sqrt{c(u)}/2$。对于同一个历史下的联合 score 向量，还可由最小二乘投影得到：

$$
\mathbf B^\top C^{\dagger}\mathbf B\le\operatorname{Var}(G\mid h)\le1/4,
\qquad \mathbf B\in\operatorname{range}(C). \tag{35b}
$$

证明：展开非负量 $\mathbb E[(G-\mathbb EG-a^\top S)^2]$，取 $a=C^\dagger B$。零特征值方向的 score 几乎处处为零，所以该方向的收益亦为零。

这个约束能帮助检查收益预测器与 Fisher 尺度是否相容。可将预测值投影到可行区间作为结构正则，但它不证明收益的正负可靠。有限样本估计违反上界可能来自噪声；只有精确条件量或有误差控制的量才能直接判定不一致。不要把不同已访问前缀的条件收益拼成同历史向量，再套式 (35b)。[本文推导]

---

## 8. 为什么同一批 rollout 既估计又选 gate 会乐观

设真实局部收益 $b(u)=0$，但单样本估计 $X$ 因回报和 token 噪声有正有负。对称例子 $X\in\{-0.5,+0.5\}$、$c=\tau=1$，直接取 $w=[X]_+$。

$$
\mathbb E[X]=0,\qquad
\mathbb E\left[Xw-\frac12w^2\right]=0.0625>0. \tag{36}
$$

**同样本 surrogate 看起来有收益，不意味着该控制器在新样本上提高环境回报。**原因不是 score estimator 本身有偏，而是先看噪声再取正值、再用同一噪声评价所引入的选择偏差。[原稿 L164–174；本文有限反例]

### 8.1 实际应采用的验证方式

R0 的作用是建立工程基线与测量问题，不为其赋予无偏门控解释。R1 使用冻结的、与被评价 prompt fold 独立的预测器来决定权重，再用该 fold 的 $X$ 评估：

$$
\widehat F_{heldout}
=\frac1N\sum_i\sum_{k,t}
\left[w(u_{i,k,t})X_{i,k,t}-\frac\tau2w(u_{i,k,t})^2c(u_{i,k,t})\right]. \tag{37}
$$

该量是旧策略访问分布上的局部 surrogate，不是有限更新后真实收益的替代品。应同时报告训练内目标与式 (37) 的差距。

### 8.2 有限样本置信下界

在固定前缀、固定方向、独立评估样本、独立有界基线等假设下，可以采用原稿的有界集中不等式得到 $\underline b$。多个方向同时选 gate，需要同时覆盖所有被选择的方向，而不是逐个 95% 区间后忽略多重选择。

对于非负 $\mathbf w$，若有同时成立的 $B_j\ge\underline B_j$ 与 $K(\mathbf w)\le\overline K(\mathbf w)$，则：

$$
\mathbf w^\top\mathbf B-\tau K(\mathbf w)
\ge\mathbf w^\top\underline{\mathbf B}-\tau\overline K(\mathbf w). \tag{38}
$$

在全支持可行域内最大化右边并保留零更新，可以给出相应条件证书。

**注意**：LOO 的各项不是相互独立样本；不能直接把它们塞进“独立样本 Hoeffding/标准误差”公式。理论审核可以改用已冻结独立基线；实际统计则应按 prompt/轨迹簇处理依赖。模型 ensemble 的标准差也不是自动成立的频率学置信界。[原稿 §12 的独立性前提；本文统计边界澄清]

---

## 9. 怎样把理想方向真正更新到 Student

这里需要区分三件事：目标 logit 方向、目标概率分布、共享参数优化后的实际方向。[原稿 L35–46、L204–208]

### 9.1 接法 A：保留原 VERPO 的 signed evidence 方向

若原 token evidence loss 在分析点满足：

$$
-\nabla_z\ell_{evi}=s,
$$

冻结 gate 后加权：

$$
\mathcal L_{evi}^{controlled}
=\mathbb E_{\tau\sim p}\sum_{k,t}m_{k,t}\,\operatorname{sg}[w_{k,t}]\ell_{evi,k,t}. \tag{39}
$$

在独立 logit 坐标，其负梯度为 $w s$。对于和为零的 signed FKL 方向，一个核对用形式是：

$$
\ell_{evi,k,t}=-\sum_v\operatorname{sg}[s_{k,t}(v)]\,z_\theta(v\mid u_{k,t}). \tag{40}
$$

这是方向实现用的线性损失，本身通常无下界，不能脱离 trust/anchor/小更新单独大步优化。生产实现优先复用已有证据损失，并用梯度核对确认其 logit 负梯度确为本文的 $s$。

如果 $s$ 含常数方向，可在合法支持上减去算术均值去掉 logits 的 gauge 分量；这不改变 softmax 微扰方向。不同的归一化和裁剪仍需要重算 $b,c$。

### 9.2 接法 B：构造显式目标分布再投影

易实现且总合法的 token 目标为：

$$
q^{soft}(v\mid u)=\frac{p(v\mid u)e^{w(u)s(u,v)}}{\sum_bp(b\mid u)e^{w(u)s(u,b)}}. \tag{41}
$$

训练：

$$
\mathcal L_{target}=\mathbb E_{u\sim d^p}
D_{KL}(q^{soft}(\cdot\mid u)\Vert\pi_\theta(\cdot\mid u)). \tag{42}
$$

若理想优化与表达能力充分，目标是逼近 $q^{soft}$。但式 (41) 在有限强度下不同于式 (22) 的完整动作线性路径。

更重要的是，在 $\pi_\theta=p$ 处：

$$
-\nabla_zD_{KL}(q^{soft}\Vert p)=q^{soft}-p
=wF(p)s+O(w^2), \tag{43}
$$

**不是 $ws$。**不能把“有限次普通 KL 梯度更新”直接说成“沿原 logit 方向走了 $w$”。这是蒸馏目标与单步优化器的区别；不是说显式目标蒸馏不可用。

### 9.3 精确线性动作目标的因果分解

若已经定义了所有前缀上的可预测强度 $\alpha(u)=\sum_jw_jf_j(u)$，令：

$$
M_{<t}=\sum_{r<t}\alpha(u_r)\zeta_r.
$$

则式 (22) 的精确 token 条件分布是：

$$
q^{aff}(v\mid u_t)
=p(v\mid u_t)
\frac{1+M_{<t}+\alpha(u_t)\zeta(u_t,v)}{1+M_{<t}}. \tag{44}
$$

未来 score 的条件均值为零，故前缀边缘概率为 $p(u_t)(1+M_{<t})$，由概率比即得式 (44)。$M$ 在每个 Agent step 开始时重置。

这不是任意把事后权重代进去即可使用：必须有全前缀定义、可预测性、共同支持与全支持正性保证。例如 $|s|\le L$、每步最多 $T_{max}$ 个 token、$\alpha\le\alpha_{max}$，则 $2T_{max}L\alpha_{max}<1$ 是一个保守充分条件。

只在 8 条已采样轨迹上检查正性，不能保证未采样动作合法。[原稿 §7；本文多方向扩展]

### 9.4 与原 reference / GRPO 分量的关系

保持语义分离：

$$
\mathcal L_{total}
=\lambda_{evi}\mathcal L_{evi}^{controlled}
+\lambda_{ref}\mathcal L_{ref}
+\lambda_{out}\mathcal L_{outcome}
+\lambda_{trust}\mathcal L_{trust}. \tag{45}
$$

这些项不是都必须同时新增。$\mathcal L_{ref}$ 是原无证据 Teacher anchor；$\mathcal L_{trust}$ 针对旧 Student 的变化；$\mathcal L_{outcome}$ 可以保留为已有 GRPO/RLOO 分支。纯 VERPO 控制实验可以先设 $\lambda_{out}=0$，把 outcome 分支作为单独消融。

**不要再用 $A_i$ 乘一次已经由 $A_i\zeta$ 计算出的纠正方向，除非明确是在实验另一种 advantage modulation 目标。**本文的 $A_i$ 用来估计方向收益，权重一旦形成，控制的是固定 $s$。

共享参数下：

$$
\delta z(u)=J_z(u)\delta\theta.
$$

所以即使 evidence 的局部 logit 梯度正确，Adam、梯度裁剪、loss normalization、其他损失与参数共享仍可能改变实际位移。**权重为零只意味着不主动施加该位置的 evidence 项，不意味着该位置分布不变。**[原稿 L204–208、L348、L494–501]

---

## 10. 端到端训练伪代码

以下是算法伪代码，不是假定某个框架已经提供这些同名函数。附带 NumPy 文件实现了统计与局部控制内核；真实模型回放、Teacher 证据构建、Agent 环境和分布式训练适配尚需在现有项目内接入。

### 10.1 R0：不加收益网络的最小版本

```python
# 固定：每个 prompt 的 M=8 条轨迹；方向未乘当前回报导出的 gate。
# 输出：事后训练权重；不是生成前已定义的有保证控制策略。

for iteration in training_iterations:
    old_student = freeze_snapshot(student)
    teacher = freeze_snapshot(teacher_or_ema)
    evidence_cfg = freeze_independent_evidence_config()
    behavior_cfg = freeze_sampling_config()  # 温度、hard mask、支持等

    # 每个任务实例独立重置环境，保留失败、EOS 和 step 边界。
    groups = rollout_agent(
        old_student, prompts, n_per_prompt=8,
        sampling=behavior_cfg, keep_failures=True,
    )

    replay_rows = []
    for group in groups:
        R = terminal_verifier(group)          # [8]，先用原始成功奖励
        A = R - (R.sum() - R) / 7.0           # 不除 reward 标准差

        for i, trajectory in enumerate(group):
            for step in trajectory.policy_steps:
                # observation 留在 prefix 中，但不作为策略输出计分。
                p, sampled_ids, policy_mask = replay_old_behavior(
                    old_student, step, behavior_cfg
                )                             # [T,V], [T], [T]
                teacher_probs = causal_teacher_replay(
                    teacher, step, evidence_cfg
                )
                s = build_raw_logit_direction(p, teacher_probs)
                s = stop_gradient(s)         # 若裁剪，下面用裁剪后的 s

                mu = (p * s).sum(-1)
                zeta = gather(s, sampled_ids) - mu
                c = (p * (s - mu[:, None])**2).sum(-1)
                zeta = where(policy_mask, zeta, 0)
                c = where(policy_mask, c, 0)
                b_sample = A[i] * zeta

                # 先挑 token 子集，再比较该子集净收益与 step_fee。
                # 不先用 sum(b_sample)<=0 删掉整步。
                w = solve_diagonal_step_surrogate(
                    b=b_sample, c=c,
                    tau=tau, cap=weight_cap,
                    linear_penalty=lambda_token,
                    step_fee=kappa_step,
                )
                w = stop_gradient(w)

                replay_rows.append(
                    store(step, p, s, b_sample, c, w, policy_mask,
                          gate_semantics="retrospective_empirical")
                )

    # 轨迹内求和，轨迹间平均；不要偷偷改成每条按长度/权重和归一化。
    loss_evi = mean_over_trajectories(
        sum_policy_tokens(w * original_signed_evidence_loss(student, s))
    )
    loss = combine_separate_losses(loss_evi, reference_loss, optional_outcome_loss)
    update_student_once_or_with_small_trust_region(loss)
    log_replay_estimation_gate_geometry_and_realized_update(replay_rows)
    # 多轮 optimizer epoch 会远离 old_student：重采样或明确离策略修正。
```

首轮控制参数建议以 $\lambda_{token}=\kappa_{step}=0$ 的最小形式跑通，再单独消融稀疏惩罚；不预设“稀疏一定更好”。$\tau$、cap、ridge 必须通过训练/验证协议决定，本报告不把旧任务超参数直接搬成 Agent 最优设置。

### 10.2 R1：加入前缀条件预测与交叉拟合

```python
# 每个 outer batch 内固定 old Student / Teacher / direction。
rows = collect_raw_replay_rows_without_gating(M=8)
# 每行含：prompt_id, trajectory_id, step_id, pretoken_features, X=A*zeta, c, s
folds = split_by_prompt_or_task_instance(rows)

for fold in folds:
    train_rows = rows_outside(fold)
    eval_rows = rows_inside(fold)

    benefit_head = fit_squared_error_predictor(
        features=train_rows.pretoken_features,
        labels=train_rows.X,
    )
    benefit_head.freeze()

    for row in eval_rows:
        # 输入禁止当前 sampled token、当前最终回报和未来 observation。
        b_pred = benefit_head(row.pretoken_features)
        row.w = clip((b_pred - lambda_token) /
                     (tau * (row.c + cost_ridge)), 0, weight_cap)
        row.w = where(row.c > fisher_floor, row.w, 0)

    # 这一 fold 用独立于其结果的预测器定义 token gate。
    # 若要求 step-start gate，用另一个冻结的 V_head(h_step)；
    # 不用本条完整 step 的事后 sum-utility 冒充因果 step-start gate。
    evaluate_fixed_gate_with_heldout_X(eval_rows)

# 用 out-of-fold weights 更新 Student；更新强度仍需实际测量。
project_or_apply_controlled_evidence(rows)
```

这里的 $c$ 和方向特征可以来自 Teacher replay，因为 Teacher 方向已独立固定；但这些特征是否能在部署时取得，属于训练到部署的另一个约束。如果仅训练时使用，则报告应称其为“训练期因果前缀 oracle/控制场”，不要宣称无 Teacher 推理时仍直接执行同一 gate。

交叉拟合也不能补救来自本 fold 的 hindsight evidence。方向数据依赖必须另外隔离。

### 10.3 小型机制审核的分支干预

```python
for anchor in heldout_restorable_histories:
    freeze_old_policy_teacher_and_candidate_controller()
    # 先固定候选：uniform / random matched budget / selected-token controller
    # 再用独立随机 continuation 测量，而非用选 gate 的样本重复评分。
    base_returns = repeat_old_student_from(anchor)
    for epsilon in predefined_small_strength_grid:
        corrected_returns = repeat_from(
            anchor,
            current_step_policy=causal_corrected_distribution(epsilon),
            later_steps_policy=old_student,
            execute_fresh_tool_results=True,
        )
        estimate_gain_and_prompt_cluster_uncertainty(
            corrected_returns, base_returns
        )
```

在固定 $h$ 和真正被定义的方向下，小强度差商 $(J_h(p^\epsilon)-J_h(p))/\epsilon$ 可核对 $B(h)$。简单换成一条 Teacher 动作再减 Student 动作的回报，通常估计的是动作替换收益，不是 FEC 整个分布方向的 $B$。[原稿 L396–404]

---

## 11. 新增诊断指标：记录什么、支持什么结论

你已有的 `WeightEffectiveCoverage`、`WeightMean`、`EvidenceDisplacementL2`、`FECResidualCov` 继续保留。它们分别描述权重覆盖、强度、原始证据位移与 nuisance 投影残差，不足以证明 Agent 的条件收益估计准确或 token 选择真正有益。以下为本报告建议新增的指标，不声称已存在于当前代码。

### 11.1 P0：正确性，不通过就不要解释机制

| 指标名 | 定义/统计对象 | 作用与边界 |
|---|---|---|
| `ReplayLogprobMAE` / `P99` | 有效策略 token 的 $\lvert\log p_{replay}(y\mid u)-\log p_{rollout}(y\mid u)\rvert$ | 检查旧策略快照、温度、模板和输出索引一致性；容差由数值精度决定，不预设必须严格为零 |
| `PrefixCausalityDelta` | 固定前缀，改变未来 suffix/终局奖励后，早期方向 $s(u)$ 的差异 | 应只剩数值误差；能抓住未来 evidence、路由和 hidden-state shift 泄漏 |
| `ObservationMaskLeakRate` | observation/padding 上非零 evidence 权重或输出监督占比 | 与 `PolicyMaskCoverage` 联合检查，避免漏算 EOS 或工具参数 |
| `CenteredScoreResidual` | $\lvert\sum_vp(v\mid u)(s(v)-\mu)\rvert$ 的均值/max | 词表内代数检查；不能证明采样分布与 p 一致 |
| `EmpiricalScoreMean` | 同历史或固定可预测场下采样 $\zeta$/$S$ 的均值与簇级不确定性 | 应统计上接近零；不要把所有 token 当独立样本构造误差条 |
| `SupportViolationRate` / `RetainedMass` | 活跃 sampled token 的 p 为零比例；Top-K 保留概率质量 | 检查所估计对象是否与真实行为支持一致 |

### 11.2 P1：收益估计是否可靠

| 指标名 | 定义/统计对象 | 作用与边界 |
|---|---|---|
| `BenefitRawSignedMean` / `NegFraction` | 原始 $X=A\zeta$、step $AS$ 的均值/负值比例 | 必须在 ReLU/门控之前记录；另记预测值，不能混列 |
| `SameHistorySampleCount` | 完整 prefix hash 下的独立续写样本数分布 | 直接展示 M=8 在中间前缀是否退化为 1 |
| `ReturnDegenerateGroupRate` | 同 prompt 所有 8 条 reward 相等的比例 | 分别报告 all-success/all-failure；说明多少组没有组内可分辨信号 |
| `BenefitPredictionCalibration` | 按 out-of-fold $\hat b$ 分桶，比较桶内预测均值与独立 $X$ 均值 | 是回归校准，不是 token 正误 ECE；MSE 同时含不可约 MC 噪声 |
| `GateGeneralizationGap` | 同样本门控 surrogate 与 held-out 固定 gate surrogate 的差 | 揭示选择乐观偏差，不能仅汇报训练内正收益 |
| `BenefitMomentBoundViolation` | 在回报 [0,1] 和完整词表下，预测 $\hat b^2>c/4$ 的比例；同历史联合版本检查式 (35b) | 结构一致性诊断，不把有噪声估计的越界自动判成代码错误 |
| `InsufficientEvidenceRate` | 有明确审核规则时，被标为证据不足而非正/负收益的比例 | 预测器标准差不能未经校准直接当置信区间 |

### 11.3 P1：step/token 选择是否真的改变了行为

令 $\varepsilon_w$ 为事先固定的有效权重阈值。为接续现有口径，可同时报告严格 $w>0$ 和既有 $10^{-3}$ 阈值下的统计，但不把两者混为一谈。

$$
\mathrm{StepEffectiveCoverage}
=\frac{\#\{\text{eligible steps}:\max_t w_{k,t}>\varepsilon_w\}}
{\#\{\text{eligible steps}\}}. \tag{46}
$$

$$
\mathrm{TokenCoverageGivenActiveStep}
=\frac{\#\{\text{active steps 内 }w>\varepsilon_w\}}
{\#\{\text{active steps 内 eligible policy tokens}\}}. \tag{47}
$$

| 指标名 | 额外记录内容 |
|---|---|
| `StepEffectiveCoverage` | step 级分母，而非 token 级分母 |
| `TokenCoverageGivenActiveStep` | 区分“选了少量 step”与“每个 step 内只改少量 token” |
| `SubsetRescueRate` | 原始 $\sum_t\hat b_t\le0$ 的 step 中，子集控制仍激活的比例；这是代理选择统计，另需真实干预收益验证 |
| `WeightCapRate` / `GateMargin` | 达到权重上界比例；$V_k^*-\kappa_s$ 分布 |
| `ZeroFisherNonzeroBenefitRate` | $c$ 近零但 $\lvert\hat b\rvert$ 显著非零的比例；用于发现预测器不一致 |
| `CorrectionByTokenRole` | reasoning / tool-name / arguments / answer / EOS 的覆盖与代价；角色切分规则须固定 |

### 11.4 P1：几何代价是否算对、目标是否实现

$$
\mathrm{FisherIdentityGap}
=\frac{\widehat{\mathbb E S^2}-\widehat{\mathbb E\sum_t c_t}}
{\max(\widehat{\mathbb E\sum_t c_t},\epsilon)}. \tag{48}
$$

这个恒等式检查只用于固定、可预测方向。事后依赖 reward 的 gate 乘进去后，不能继续要求它必然成立。

| 指标名 | 定义/统计对象 | 不可越过的边界 |
|---|---|---|
| `FisherIdentityGap` | 式 (48)，固定历史/固定方向或明确的全局场 | 单样本差异正常；需看重复与样本量趋势 |
| `TokenKLQuadraticGap` | 在已访问前缀比较精确 $KL(q^{soft}\Vert p)$ 与 $\frac12w^2c$ | 仅是条件 token KL 的近似检查，不等于完整 step KL |
| `OldPrefixConditionalKL` | 在旧策略访问前缀测量更新后 $KL(\pi_\theta\Vert p)$ | 不可直接命名为新策略完整 step KL |
| `TargetProjectionKL` | 目标 $q$ 与更新后 Student 在审核前缀的 KL | 衡量实现误差，但不能代表未访问历史的误差上界 |
| `RealizedDirectionCosF` | 实際 $\delta z$ 与目标 $ws$ 的 Fisher 夹角 | 捕捉双重映射、共享参数、其他 loss 的影响 |
| `OffRouteDrift` | 权重为零位置及未训练审核前缀的 KL/准确率变化 | 不应预设为零；anchor/outcome 分量本身也会更新这些位置 |

$$
\mathrm{cos}_F(a,b)=\frac{a^\top F(p)b}
{\sqrt{a^\top F(p)a}\sqrt{b^\top F(p)b}+\epsilon}. \tag{49}
$$

若梯度空间与 logits 空间尺度不一致，首先比较方向，再通过实际 $\delta z$ 测量幅度。不要未经 learning-rate/梯度尺度校准，把 $w^2c/2$ 宣称为训练后实际 KL。

### 11.5 P2：是否真正有环境收益

| 指标名 | 审核方式 |
|---|---|
| `HeldoutLocalInterventionGain` | 固定方向与 gate 后，在相同可回放历史上独立执行受控分布，后续旧 Student 继续 |
| `SelectionGainOverRandom` | 与相同接受数、相近 Fisher 移动预算的随机 token/step 选择比较 |
| `StrengthDoseResponse` | 对预设小强度网格测量实际回报、KL、无效动作率，验证 $b/c$ 排序是否对应可利用增益 |
| `SubsetRescueInterventionGain` | 对“原整步负收益但局部子集被选中”的历史专门验证 |
| `GradientConflict` | evidence/reference/outcome 参数梯度两两夹角及范数，作为共享参数解释，不作为因果性能证据 |

### 11.6 Agent 任务与成本指标

保留主任务成功率，同时新增/分层报告：合法工具调用率、循环/重复调用率、提前终止率、工具失败后的恢复率、平均工具调用数、生成 token 数、每成功任务的环境交互与 Teacher replay 成本。

Teacher token 前向数、不同 evidence 条件的 replay 次数、旧 Student replay 成本、收益预测器成本、额外分支审核成本应分别统计。**门控稀疏不等于训练或教师计算更便宜。**

### 11.7 最小必需面板

首版至少保留：回放一致性、因果泄漏、mask、原始有符号收益、奖励退化组比例、同历史样本数、step 覆盖、step 内 token 覆盖、Fisher 恒等式差距、独立 gate 评价差距、实际方向/投影误差、等预算成功率。其他指标可在管线稳定后增补。

---

## 12. 原讨论尚未充分覆盖的风险与应对

### 12.1 每个 token 都有分数，不代表每个 token 都有足够证据

8 条轨迹可能只有首个 token 真正共享前缀，甚至“第一步内第 3 个 token”也早已分叉。一个 prefix score 不能当作 8 样本局部因果标签。需要条件回归、明确的前缀抽象，或少量同历史审核。

### 12.2 hindsight 训练与非预知策略不是同一对象

监督训练可以利用答案；但要把理论中的策略改进保证移到一个 Teacher-conditioned 目标上，必须说明 Teacher 信息是否为可见历史的函数。若依赖隐藏变量 $e$，可定义 $q(v|u)=\mathbb E[q(v|u,e)\mid u]$ 作为合法目标，但这个边际化本身不一定便宜，单个隐藏状态样本不等于已完成边际化。[原稿 L70、L80、L350]

### 12.3 训练权重不一定是因果控制强度

$w$ 若只是 evidence loss 系数，真正 $\delta z$ 还取决于优化器、批次归一化、学习率、梯度累积、LoRA 子空间及其他损失。需要实际方向和投影测量，不只画权重热力图。

### 12.4 不能一边用全未来确定语义 span，一边把 span gate 当作可预测

“这一整句是错误推理”或“这个 span 最终导致失败”可能只有生成后才知道。训练时可做事后 span 选择，但部署或精确策略场需要由前缀定义的边界规则。若多个 token 共享一个 span 强度，其方向是 $S_{span}=\sum_{t\in span}\zeta_t$，但条件收益与跨度抽样仍需按相应 estimand 处理。

### 12.5 奖励稀疏与全失败/全成功组

组内奖励一致时 LOO 信号为零，不证明 $b=0$。不能让 Teacher 置信度伪装成“已经验证的任务收益”。可以保留独立探索分支、扩大某些困难状态采样、引入经过验证的进度奖励，或使用价值/收益模型；这些都是新增设计，需要单独隔离。[原稿 L462–466]

### 12.6 方向本身可能覆盖不到有益行为

在当前方向子空间内最优，不等于整个策略空间最优。Teacher 给错方向、参考答案格式不适配 Student、关键工具行为在采样支持外时，即使控制器完美也可能无增益。负收益表示拒绝该方向，不表示该状态不可学。

### 12.7 有限强度的多 token 协同与反作用

零点 Fisher 正交不消除有限更新的语义协同。工具名和参数必须联合改变；单独修改 JSON 引号、工具名或答案单位可能破坏可执行性。初步可以把工具调用的结构化字段作为固定方向组，对 token 独立控制做 span/group 对照，但不能把语义组当成已证明正确的信用单位。

### 12.8 环境干预必须生成新的 observation

真实干预改变工具调用后，不能仍接回原轨迹缓存的旧 observation。否则评估的是不一致的轨迹，不是 $Q^p$。真实账户/生产工具中，审核应使用可恢复沙箱、只读任务或 mock 环境，避免重复付费、发信、写入等不可逆副作用。[原稿 L398–404；工程安全要求]

### 12.9 非平稳环境与伪“同状态”

相同 prompt 文本不代表环境状态相同；网页内容、数据库、工具版本、网络失败和隐藏随机种子都可能变化。记录 task instance/environment version，必要时保存可恢复快照。隐藏状态不由历史唯一决定时，精确条件值需要对兼容隐藏状态积分。

### 12.10 训练多 epoch 的离策略问题

收益与方向在旧 $p$ 下定义。更新几轮后继续用同一批轨迹和方向，会偏离分析点。应限制更新、监控 KL、重新采样，或明确构造重要性修正；截断 IS 又引入偏差。不能用“on-policy rollout 是刚采的”覆盖后续所有 epoch 的离策略漂移。

### 12.11 信号尺度与跨任务比较

若方向缩放 $s\mapsto as$，则 $b\mapsto ab,c\mapsto a^2c$。不含 cap/额外惩罚时 $w^*\mapsto w^*/a$，理想位移 $w^*s$ 不变；固定 cap、ridge 和幅度惩罚会破坏该不变性。

奖励缩放 $R\mapsto qR$ 会使收益缩放而 Fisher 不变；要保持同一控制，$\tau$ 与相关收益单位的罚项也需相应缩放。不同任务的奖励尺度、长度和工具成本要分别记录，不能只比原始 $B/C$。

### 12.12 减少 Teacher 使用的时机问题

若为每个 token 先跑完所有 evidence Teacher，再挑出 5% token，可能节约更新预算，却没有节约这些 Teacher 前向。要减少 Teacher replay，必须有更早的 cheap gate；它自身的错拒率、漏掉的收益和计算开销要成为新实验。

### 12.13 Top-K、低精度与 tokenizer 对齐

Teacher 与 Student 的词表不一致时，$q^+-q^-$ 不能按数组下标相减。相同 tokenizer 也要保证特殊 token、logit mask、模板和预测 shift 一致。长序列 BF16 的 $E[s^2]-E[s]^2$ 可能有消减误差，优先用 $E[(s-\mu)^2]$ 并用 FP32/更高精度累计核对。Top-K 必须保留质量和误差日志。

### 12.14 奖励审核本身可能出错

可执行 verifier、LLM judge、格式正则等各有语义。错误答案被接受、工具失败被算成成功、长回答钻奖励规则都会污染 $b$。保留失败原始轨迹和独立任务审计；方向与 judge 共用模型也不能当作独立证据。

### 12.15 超时和截断不是统一的“失败”

任务若明确把超时判失败，可以纳入终局奖励；若只是收集系统提前截断，则真实未来回报被删失。直接给 0 改变了目标，需要单独标记，或引入带误差说明的 bootstrap。

### 12.16 样本相关性与统计检验

同 prompt 的 8 条轨迹、同轨迹的多个 token、同批 LOO 项都存在结构性关联。成功率、诊断置信区间和显著性检验应以任务实例/prompt 为主要簇单位；不能拿数百万 token 当数百万独立试验。

---

## 13. 从局部纠正到整体 Agent 收益：能证明到哪里

对任意同环境的新策略 $q$，有限时域 performance-difference 恒等式是：

$$
J(q)-J(p)=\sum_k\mathbb E_{h_k\sim d_k^q}
\left[\sum_aq_k(a|h_k)A_k^p(h_k,a)\right]. \tag{50}
$$

证明：用 Bellman 等式展开 $A_k^p$，在新策略轨迹上求和，中间价值项望远镜相消。关键是历史分布为 **新策略 $d^q$**，不是无代价替换成旧访问分布。[原稿 §10；外部基础 S2]

若每个相关历史上都精确实现式 (22)，且选择满足 $\mathbf w^\top\mathbf B\ge0$，则整体不下降。这个条件需要精确/有证书的收益、全支持可行目标、非预知策略和所有相关历史上的实现。

若实际训练得到 $\widetilde p$，目标为 $q$，令：

$$
\epsilon_k(h)=TV(\widetilde p_k(\cdot|h),q_k(\cdot|h)).
$$

则沿用原稿的论证：

$$
J(\widetilde p)-J(p)
\ge\sum_k\mathbb E_{h\sim d_k^{\widetilde p}}
\left[\mathbf w_k^\top\mathbf B_k
-\operatorname{span}(Q_k^p(h,\cdot))\epsilon_k(h)\right]. \tag{51}
$$

**目标收益必须覆盖投影误差，而且误差要在实际新访问历史上受控。**训练集平均 token KL 小，不等于满足式 (51) 的全部条件。加入其他损失的实际参数更新也不能自动继承这个定理。

本报告的 R0/R1 只是通往上述条件的可实施路线，尚未证明它们在真实神经网络中满足全部条件。

---

## 14. 推荐实验矩阵与实施顺序

### 14.1 阶段 A：冻结模型，先核对估计器

使用同一旧 Student/Teacher 和固定证据，收集 M=8 的真实轨迹。只做 replay，不更新模型。检查采样概率、方向坐标、mask、原始收益、条件样本数和 Fisher 分解。选取少量安全可回放历史，额外估计独立小强度干预收益。

验收：代数和 replay 一致性不出错；预测/估计收益与独立干预至少具有可量化的一致性，或明确暴露失败机制。不以“多数样本 $X>0$”作为通过标准。

### 14.2 阶段 B：训练消融，先隔离控制作用

| 对照 | 保持什么相同 | 检验什么 |
|---|---|---|
| 无 evidence correction | Student/任务/基础训练预算 | 纠正本身的增益 |
| 固定教师方向、统一强度 | Teacher 和方向 | 自适应强度是否必要 |
| 只做标量 step 控制 | 相同原始 token 方向 | token 子集自由度的价值 |
| R0 token/step 经验控制 | 相同 rollout 与回放 | 低成本经验实现的可用性与选择偏差 |
| R1 前缀条件控制 | 相同任务/方向；计入预测器成本 | 条件建模是否改善独立选择收益 |
| 随机位置 + 匹配移动预算 | 接受数、预期 Fisher/实际 KL | 选择位置是否比纯稀疏正则更有效 |
| 原平滑 ZPD vs 二次解 | 原始方向、相近实际移动预算 | 控制函数形状而非强度差异 |
| 加/不加 outcome 分支 | reference/evidence 设置 | 新增收益是否其实来自 GRPO/RLOO |
| 干净 evidence vs hindsight evidence | 明确独立性与计算预算 | 强信号是否源于未来信息与部署不匹配 |

不要一次同时改 Teacher、证据构造、FEC、gate、奖励、rollout 数和损失，随后把收益全部归因于联合控制。

### 14.3 阶段 C：Agent 专属压力测试

检查工具返回扰动、可恢复失败、较长 horizon、多次重试、错误早停以及新访问历史。只在有安全可回放环境时做干预；真实不可逆工具的评估不能靠反复执行产生风险。

### 14.4 公平预算

分别报告固定环境交互量、固定 Teacher 前向量、固定总计算成本下的结果。额外分支采样、预测器拟合、K 条 evidence replay 都计入成本。训练验证与最终测试保持任务实例隔离。

---

## 15. 工程改动清单

建议按模块改动，而不是立即把所有逻辑塞进原有 token weight 函数。

| 模块 | 新增职责 | 必需验收 |
|---|---|---|
| Agent rollout 数据层 | 完整 history、step 边界、动作/observation/EOS mask、版本信息 | 原轨迹能准确重放；无效动作未被删除 |
| Replay adapter | 对齐旧 Student 行为概率与 Teacher 各证据分布 | logprob 一致性、未来替换不影响早期方向 |
| Direction adapter | 返回明确坐标的 raw $s$，不含当前 outcome gate | 数值有限差分验证 logit 方向 |
| Statistics kernel | $\zeta,c,X,S$、LOO 与条件组信息 | 与附带 NumPy 核心及有限例子一致 |
| Joint controller | diagonal surrogate；可选低维 full-C QP；zero fallback | 与目标函数数值网格/KKT 对照 |
| Optional benefit head | 前缀特征、prompt 级 fold、预测校准 | 无当前 token/未来泄漏；独立收益评价 |
| Loss integration | stopped weights、reference/evidence/outcome 分离 | 不重复乘 advantage，不压平 step 差异 |
| Update auditor | 目标/实际方向、off-route drift、KL | 学习率和多 loss 影响有日志 |
| Intervention evaluator | 固定候选、真实新 observation、独立 continuation | 评估对象是分布方向，而非偷换 Teacher 单动作 |

原稿提到的“取每行首个 advantage 再广播”是一个需要重点核查的旧入口，但本次未读取当前仓库，不能断言该行为至今仍未变化。接入时应重新核查数据流。[原稿 L386、L494–499]

---

## 16. 随附代码与本次实际核对

### 16.1 文件

- `agent_verpo_core.py`：NumPy 参考内核，含 token 统计、LOO、同历史估计、带 step 激活费的对角控制、softmax 目标、精确 affine 前缀目标及有限动作联合目标/梯度/Hessian。
- `verify_agent_verpo.py`：有限分布与代数核对脚本。
- `verification_results.json`：本次实际运行输出。

运行：

```bash
python verify_agent_verpo.py
```

依赖为 NumPy；不需要 Teacher API、GPU 或真实 Agent 环境。这些代码不包括生产模型训练器、FEC 业务实现或 Agent 工具适配。

### 16.2 结果

本次 **17 项有限核对全部通过**，包括：

1. 原稿两 token 有限例子的归一化、零均值、$B=0.0888,C=0.38976$。
2. softmax 方向与价值导数的有限差分。
3. Fisher chain rule 与零点跨 token 正交。
4. 多方向 affine 目标的精确前缀分解。
5. 联合梯度/Hessian 的有限差分与有限强度交叉项。
6. M=8 手算例子的 LOO 与 Fisher。
7. 枚举全部 $2^8$ 个 Bernoulli 样本组，核对 LOO 无偏和包含自身均值的 $7/8$ 缩放。
8. 原整步负收益、局部子集有益的 rescue 例子与 step 激活费用。
9. 对角控制闭式解与密集数值网格。
10. 无额外约束时的方向/奖励尺度关系。
11. 事后 gate 破坏零均值与同样本选择乐观反例。
12. 行为采样分布不一致导致 score 均值非零的反例。
13. tilted KL 梯度多一层 Fisher 映射。
14. observation/padding mask。
15. 非法支持和不可行 affine 概率被拒绝。
16. 两步有限 MDP 的 performance-difference 恒等式与新 occupancy。
17. 有界回报的收益–代价矩约束与可实现的 token 子集 rescue 例子。

这些测试没有检查真实 LLM 的收益预测精度、训练稳定性、置信区间覆盖率、教师成本或最终 Agent 成功率。本次也没有修改或运行用户仓库中的训练代码。

---

## 17. 可以写入方法稿的贡献表述与不能写的主张

### 17.1 可以作为待验证方法贡献的表述

> 我们将教师证据方向的构造与其在 Agent 中的接受分离。旧 Student 的交互回报用于估计方向的后续价值收益，因果前缀回放提供 token 级方向分数与 Fisher 移动代价。在此基础上，通过联合控制选择局部纠正子集及接受强度，并显式检验有限样本选择误差与共享参数实现误差。

这是一种方法定位，不是全面查新结论。标准 score identity、Fisher chain rule、KKT、RLOO、旧 Student continuation 或 performance-difference identity 本身不能单独作为新颖性主张。[原稿 L574–584]

### 17.2 现阶段不应写成结论

- “8 条 rollout 精确识别所有 step/token 的好坏。”
- “只要 $\widehat b>0$ 就保证真实收益为正。”
- “token Fisher 在零点正交，所以所有联合更新都互不干扰。”
- “权重稀疏，因此 Teacher 计算节省同样比例。”
- “蒸馏损失小，所以全局 Agent 回报必然提高。”
- “本文 gate 是原平滑 ZPD 公式的精确最优性证明。”
- “当前已经完成了 Agent 训练实验或全面相关工作查新。”

---

## 18. 最终建议

**先把原始方向、收益估计、控制器和 Student 更新四件事分开实现。**

第一版不要求新增 Teacher rollout，也不强制新增 Q 网络：M=8 的 Student 真实交互，加因果前缀回放，即可得到 $X=A\zeta$ 与 $c$，运行 R0 并建立完整日志。但论文主线若要强调“可靠选择哪个 step/token”，应至少配合 R1 的独立前缀预测或同历史审核，报告选择偏差和实际方向误差。

算法重点不应只是“又生成了一组更细的权重”，而是证明这条链在数据上成立：

$$
\boxed{
\text{真实回报支持的方向收益}
\;\longrightarrow\;
\text{独立可验证的子集与强度选择}
\;\longrightarrow\;
\text{Student 实际实现的分布改变}
\;\longrightarrow\;
\text{等预算 Agent 收益}
}
$$

---

## 19. 来源与可追溯入口

**S0｜用户上传原稿**\
`Pasted markdown(20260910-064456).md`，全文 627 个可见文件行号。关键位置：§3 L52–83（过程与假设）；§4–6 L104–208（score、收益、Fisher）；§7–9 L214–318（线性路径、KKT、近似）；§10 L324–366（整体收益与实现误差）；§11–12 L372–466（估计、依赖、置信与稀疏回报）；§13 L470–486（多方向控制）；§14 L488–501（原稿记录的源码对应）；§16 L548–570（验收与预算）。本报告没有重新核对原稿中的相对路径源码。

**S1｜RLOO 基础**\
Ahmadian et al. (2024), *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*, arXiv:2402.14740，§2.3。支持多样本 leave-one-out baseline 的基本机制；本文的方向投影及 Agent 联合控制另行推导。读取入口：`https://arxiv.org/html/2402.14740v2`。

**S2｜策略改进与实用近似的基础**\
Schulman et al. (2015), *Trust Region Policy Optimization*, ICML / PMLR 37。用于区分理论策略改进与实际近似算法，不意味着本方案自动继承 TRPO 定理。官方论文入口：`https://proceedings.mlr.press/v37/schulman15.html`。

**S3｜本次可执行核对**\
同目录的 `agent_verpo_core.py`、`verify_agent_verpo.py`、`verification_results.json`。只支持第 16 节列出的有限数学/数值检查。
