# 从 VERPO-ZPD 到 Agent VERPO：证据方向、收益控制与 Step–Token 联合优化的完整推导

> 上游研究快照，来源提交 `9c01f2bcd05aea1b3741ba60ede9aad201e0875c`。历史候选中的 FEC、step gate 或收益预测器不覆盖[当前设计](../current_design.md)。原文与导出文件指纹见 [provenance.json](provenance.json)。

版本：2026-09-10。本文将仓库原有 VERPO-ZPD 推导、当前论文与实现、Agent 多步推导串成一份可独立阅读的中文稿。数学从定义开始展开；训练协议和代码保持原状。

> 后续讨论见[Observation 特权信息与整轨迹回放方案](Agent_VERPO_%E8%AE%A8%E8%AE%BA%E6%80%BB%E7%BB%93%E4%B8%8EObservation%E5%9B%9E%E6%94%BE%E6%96%B9%E6%A1%88.md)：可以不使用 FEC，将未来反馈放在 Teacher 前缀中进行整轨迹回放。该 hindsight 方案与本文可预测方向定理的假设不同，不能直接继承其收益保证。

## 阅读路线与来源

原来的 Agent 稿以“给定一个 Teacher 方向”为起点。这个起点省略了 VERPO 最关键的一段：为什么需要证据方向，它如何从蒸馏损失里分离出来，为什么 FKL 与 RKL 的方向不同，以及 ZPD 凭什么接受或拒绝这个方向。本文先回答这些问题，再进入 Agent。

全文的依赖顺序是：

**蒸馏混合了哪些作用 → 分离 reference 与 evidence → 构造 Fixed／CTR／FEC 方向 → 推导原版 token ZPD → 写出 VERPO 完整目标 → 引入环境历史与后续价值 → 从 token 累积到 step → 联合选择位置与强度 → 用 rollout 估计 → 更新 Student 并核对整体收益。**

| 阅读部分 | 回答的问题 |
|---|---|
| 第 1–5 节 | Teacher 为什么能提供方向，方向如何由 FKL／RKL 和 FEC 确定？ |
| 第 6–8 节 | 原版 ZPD 权重如何得到，它与实际训练目标是什么关系？ |
| 第 9–12 节 | 单段回答推广到 Agent 后，收益和代价如何重新定义并逐步推导？ |
| 第 13–15 节 | 整步控制与 token 选择如何成为同一个优化问题？ |
| 第 16–19 节 | 8 条 rollout 能估计什么，怎样训练，整体收益能保证到哪里？ |
| 第 20 节及附录 | 怎样落地核对，旧符号如何对应，哪些部分已有实现？ |

主要依据为仓库内以下材料，而非重新提出一套与原 VERPO 无关的方法：

- **[S1]** [原 VERPO-ZPD 完整推导](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/Math/VERPO_ZPD_FKL_RKL_Complete_Derivation_zh.md)：FKL／RKL 分解、六种方向、Jacobian 近似、原平滑控制器与全局系数。
- **[S2]** 当前论文的 FKL、RKL、ZPD、FEC 章节：精确梯度身份与建模假设。
- **[S3]** `risk_aware_opsd/verpo_zpd.py`：当前方向统计、损失与权重实现。
- **[S4]** [原 Agent 多步推导](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/agent%20opsd/derivation.md)：完整动作方向、局部 KL 优化及多步价值恒等式。
- **[S5]** [Agent 联合控制报告](Agent_VERPO_%E8%81%94%E5%90%88%E6%8E%A7%E5%88%B6%E7%A0%94%E7%A9%B6%E4%B8%8E%E5%AE%9E%E7%8E%B0%E6%8A%A5%E5%91%8A.md)：联合优化、估计偏差、R0／R1 和有限数值内核。
- **[S6]** 仓库 `AGENTS.md`：当前方向定义、Teacher 调度、group gate 和词表近似合同。

本文将**恒等式**、**局部近似**、**控制器设计**、**待验证的 Agent 接入**分开表述。旧文档包含历史实验配置；本文不把它们当作当前训练授权，也不复述旧实验数值为当前结果。本文源码核对基于 `0a4f8ba` 及本地公式排版修复。

## 0. Notation：符号总表

这部分可作为全文的查阅页，不需要一次记住所有符号。主线先记五个量：**$s$ 给出纠正方向，$\zeta$ 衡量采样 token 沿该方向的相对概率变化，$b$ 衡量预期收益，$c$ 衡量局部移动代价，$w$ 决定接受强度。**下面按出现的数学对象分组，并标明类型和定义位置。

记号约定：粗体表示联合控制向量；帽子表示估计量；星号表示相应指定目标的最优解；波浪号表示候选解或拟合后的策略，具体由下表区分。下标通常定位 rollout、step 或 token，上标 `F`／`R` 表示 FKL／RKL，`task`／`nuis`／`FEC` 表示方向成分。

### 0.1 数据、索引与交互历史

| 符号 | 含义与类型 | 定义位置／阅读提示 |
|---|---|---|
| $x$ | 输入问题或任务实例 | 第 2 节；同一 prompt 的轨迹共享 $x$ |
| $i,\ M$ | rollout 索引；同一 prompt 的轨迹数 | 第 6、16 节；例子中 $M=8$ |
| $k,\ H$ | Agent step 索引；最大交互步数 | 第 9 节；$H$ 单独出现是时域长度 |
| $t,\ T_k,\ T_{\max}$ | step 内 token 索引；第 $k$ 步长度；每步长度上界 | 第 9、17 节；不含 observation 和 padding |
| $\mathcal V,\ v$ | 词表集合；候选 token | 式 (N1)；向量的词表维度为 $\lvert\mathcal V\rvert$ |
| $y_{i,k,t},\ e_y$ | 实际采样 token；该 token 的 one-hot 词表向量 | 式 (Z7)、(A2)；局部分析时省略部分下标 |
| $a_k,\ o_k$ | 完整策略动作／生成序列；环境或工具 observation | 式 (A1)–(A2)；$a_k$ 包含策略生成的 reasoning、工具参数与 EOS |
| $h_k$ | 执行第 $k$ 步之前的可见交互历史 | 式 (A1)；是序列／状态对象，不是标量收益 |
| $u_{k,t}$ | 当前 token 生成前的完整前缀 $(h_k,y_{k,<t})$ | 式 (A1)；单段回答时为 $(x,y_{<t})$ |
| $j,\ d$ | 联合方向基底索引；方向基底数量 | 式 (J1)；$d$ 在此为整数，区别于带下标的方向 $d_t$ |

### 0.2 Student、Teacher 与概率路径

| 符号 | 含义与类型 | 定义位置／阅读提示 |
|---|---|---|
| $\theta,\ \bar\theta,\ \phi$ | Student 参数；局部分析基点；Teacher 参数 | 第 2、6 节；$b_\phi$ 中的 $\phi$ 另指收益预测器参数，模型身份由函数名区分 |
| $z_\theta(u)$ | Student 的词表 logits 向量 | 式 (N1)；未经过 softmax |
| $\pi_\theta(\cdot\mid u)$ | 更新中的 Student 条件概率分布 | 式 (N1) |
| $p,\ p_{\mathrm{old}},\ p_b$ | 局部 Student 分布；旧策略；实际行为采样分布 | 第 2、6、20 节；Agent 分析中的 $p$ 冻结，不能默认等于更新后的 $\pi_\theta$ |
| $E^0,\ E^+,\ E^-,\ E^e$ | 无证据、正确证据、错误证据、一般有证据条件 | 式 (N2)；是 Teacher 输入条件 |
| $q^0,\ q^+,\ q^-,\ q^e$ | 上述条件对应的 Teacher 词表分布 | 式 (N2)；固定 Teacher 与目标前缀，改变证据条件 |
| $q^{\mathrm{ref}}$ | 独立的 reference anchor 分布 | 式 (F4)、(R3)；本文默认使用 $q^0$ |
| $q_b,\ q_e$ | 当前 Teacher 路径的 baseline／target 端点 | 第 2–4 节；CTR 的 $q_b=q^-$，不等于 reference anchor |
| $q_w,\ q_w^{\mathrm R}$（第 3–4 节） | 算术／几何 Teacher 插值分布 | 式 (F2)、(R1)；用于推导 evidence loss |
| $q_w,\ q_{\mathbf w}$（第 12–13 节） | 受控 Student 的完整动作分布 | 式 (C1)、(J2)；虽复用字母 $q$，对象已变成 Student 动作策略 |
| $q^{\mathrm{soft}},\ q^{\mathrm{aff}}$ | token softmax 倾斜目标；完整动作 affine 目标的 token 条件分布 | 式 (U1)、(U4)；有限强度时通常不同 |
| $q,\ \widetilde p$（第 18 节） | 理想新历史策略；训练实际得到的 Student 策略 | 式 (P2)–(P4)；用于区分目标与实现误差 |
| $T,\ Z(w),\ \psi(\varepsilon),\ H(q)$ | 采样温度；几何路径配分函数；log 配分函数；分布熵 | 第 4、20 节；$T$ 不是动作长度，$H(q)$ 不是交互步数 $H$ |

### 0.3 证据方向与 Fisher 几何

| 符号 | 含义与类型 | 定义位置／阅读提示 |
|---|---|---|
| $\Delta$ | FKL 的 signed 证据概率差，词表向量 | 式 (F1)；作为 Student evidence loss 的 **logit 下降方向**；附录式 (B1) 的 $\pi\in\Delta$ 则复用 $\Delta$ 表示概率单纯形集合 |
| $r$ | RKL 的 log-density contrast，词表向量 | 式 (R1)；$r=\log q_e-\log q_b$，区别于环境标量奖励 $r_k$ |
| $F(p)$ | categorical Fisher／softmax Jacobian，词表方阵 | 式 (G1)；$F(p)=\operatorname{diag}(p)-pp^\top$ |
| $\langle a,b\rangle_{F(p)},\ \|a\|_{F(p)}^2$ | Fisher 内积与平方半范数 | 式 (G2)、(E6)；分别为 $a^\top F(p)b$ 和 $a^\top F(p)a$ |
| $\xi$ | RKL 的实际单位 logit 下降方向，词表向量 | 式 (R4)；$\xi=F(p)r$ |
| $s(u,\cdot),\ s_t$ | FKL／RKL 统一的单位 logit 纠正方向 | 式 (R5)；FKL 取 $\Delta$，RKL 取 $\xi$ |
| $F(p)s$ | 沿 $s$ 移动诱导的一阶 Student **概率**位移 | 式 (G3)；与 $s$ 的坐标职责不同 |
| $\Delta^{\mathrm{task}},\ \Delta^{\mathrm{nuis}},\ \Delta^{\mathrm{FEC}}$ | FKL 的任务、共同证据成分与投影残差 | 式 (E5)–(E7)；均为词表向量 |
| $r^{\mathrm{task}},\ r^{\mathrm{nuis}},\ r^{\mathrm{FEC}}$ | RKL 的对应 log-density 对比与残差 | 式 (E8)–(E9)；先映射到 $\xi$ 再计算投影 |
| $\xi^{\mathrm{task}},\ \xi^{\mathrm{nuis}},\ \xi^{\mathrm{FEC}}$ | 上述 RKL 对比经 $F(p)$ 映射后的 logit 方向 | 式 (E8)–(E9) |
| $\alpha_{\mathrm F},\ \alpha_{\mathrm R},\ \epsilon_{\mathrm{fec}}$ | FEC 的标量投影系数；ridge 稳定项 | 式 (E6)–(E9)；不是接受权重 $w$，也不是全局 loss 系数 |
| $K$（负证据） | 用于平均 $q^-$ 的负证据数量 | 式 (E3)；区别于后文 KL 函数 $K(w)$ |

### 0.4 回报、优势与 token／step 统计量

| 符号 | 含义与类型 | 定义位置／阅读提示 |
|---|---|---|
| $R_i,\ \bar R$ | 第 $i$ 条轨迹的标量奖励；同组奖励均值 | 式 (Z5)；纯终局奖励时可作为未结束 step 的剩余回报 |
| $r_k,\ G_k$ | 第 $k$ 步即时奖励；从该步开始的 return-to-go | 式 (A3)；$G_k=\sum_{j=k}^Hr_j$ |
| $J(p),\ J_{\mathrm{RL}}(\theta)$ | 策略期望总回报；参数化的 RL 收益目标 | 式 (A3)、(Z2)；函数自变量决定讨论策略空间还是参数空间 |
| $Q_k^p(h,a),\ V_k^p(h),\ A_k^p(h,a)$ | 旧 Student 的动作价值、历史价值、真实动作优势 | 式 (A3)；$A_k^p=Q_k^p-V_k^p$ |
| $Q^p(u,v)$ | 固定 token 前缀和候选 token 后的继续价值 | 式 (T3)；包含当前动作剩余生成与后续环境交互 |
| $\widehat A_i,\ A_i^{\mathrm{LOO}}$ | GRPO 组标准化优势；留一法原始尺度优势 | 式 (Z5)、(I1)；两者尺度及统计性质不同 |
| $g_t^{\mathrm{RL}}$ | sampled-token RL 的局部 logit 上升方向，词表向量 | 式 (Z7)；$\widehat A_i(e_y-p)$ |
| $\zeta(u,y),\ \beta(u)$ | 中心化方向 score，标量；合法的回报基线 | 式 (T2)–(T3)；$\zeta=s(y)-\mathbb E_ps$，不是 token 正误标签 |
| $\widehat b_t,\ h_t$ | 单样本 signed benefit 代理；其非负部分 | 式 (Z9)、(Z11)；$h_t=[\widehat b_t]_+$，区别于历史 $h_k$ |
| $b(u),\ c(u)$ | token 条件期望收益；条件 Fisher 移动代价，均为标量 | 式 (T3)–(T4)；$b$ 是条件真值，不能与单样本 $\widehat b$ 混用 |
| $S_k(a),\ B_k(h),\ C_k(h)$ | step 总 score；完整动作方向收益；step Fisher 代价 | 式 (S1)–(S4)；$S_k=\sum_t\zeta_{k,t}$ |
| $d_k(a)$ | 完整动作概率空间的零质量方向 | 式 (S1)；$d_k(a)=p_k(a\mid h)S_k(a)$，不是参数空间方向 $d_t$ |
| $X_{i,k,t},\ Y_{i,k,t}$ | 收益学习样本 $A_i^{\mathrm{LOO}}\zeta_{i,k,t}$；对应 Fisher 值 | 式 (I4)；条件估计的数据，不是无噪声真值 |
| $\widehat B,\ \widehat C,\ b_\phi(f(u))$ | 有限样本 step 统计估计；前缀收益预测器 | 式 (I3)、(I5)；$f(u)$ 是生成当前 token 前的特征 |

### 0.5 权重、联合优化与参数更新

| 符号 | 含义与类型 | 定义位置／阅读提示 |
|---|---|---|
| $D_t,\ d_t$ | logit Jacobian；单位 evidence 参数方向 | 式 (Z1)；$D_t=\partial z_t/\partial\theta$，$d_t=D_t^\top s_t$ |
| $\eta,\ \kappa_t,\ \delta\theta,\ \delta z$ | 有效参数步长；局部各向同性比例；实际参数／logit 变化 | 式 (Z2)、(Z8)、(U5)；$\kappa_t$ 不是 step fee $\kappa_s$ |
| $\mathcal A_t,\ \mathcal C_t$ | 参数空间真实一阶方向收益；局部负 reward 曲率 | 式 (Z3)；不同于 advantage $\widehat A_i$ 和 Fisher cost $c_t$ |
| $w_t^{\mathrm{old}},\ w^{\mathrm{quad}},\ w^\star$ | 原平滑权重；二次模型解；指定目标最优权重 | 式 (Z4)、(Z12)、(C4)；星号只对应所在目标，不表示普遍最优 |
| $a_c,\ \epsilon_c,\ d_t^{\mathrm{cost}}$ | 原控制器的 Fisher 成本尺度、正 floor、合成分母项 | 式 (Z11)；$d_t^{\mathrm{cost}}=a_cc_t+\epsilon_c$ 为标量 |
| $c_0,\ \beta,\ \rho$ | 原文旧版成本 floor、Fisher 比例、额外 floor | 第 6.3 节；$a_c=\tau\beta$，$\epsilon_c=\tau(c_0+\rho)$；此处 $\beta$ 不是回报基线 $\beta(u)$ |
| $\tau,\ \bar w,\ \bar w_j$ | KL／二次移动惩罚系数；标量或坐标接受上界 | 式 (C4)、(D2)；不是采样温度 $T$ |
| $f_j(u),\ S_j(a),\ \mathbf S(a)$ | 可预测方向基底；基底累计 score；联合 score 向量 | 式 (J1)；$f_j$ 是方向基底，区别于预测特征 $f(u)$ |
| $\mathbf w,\ \mathbf B,\ C_{ij}$ | 联合权重、联合方向收益向量；score 二阶矩矩阵 | 式 (J1)–(J4)；对角近似中 $C_j$ 表示 $C_{jj}$ |
| $\mathcal W,\ K(w),\ K(\mathbf w)$ | 联合可行域；完整动作新旧分布间的 KL 函数 | 式 (C3)、(J3)；$K(\cdot)$ 不是负证据数量 |
| $g_j$ | 联合目标对权重坐标的导数 $B_j-\tau\partial_jK$ | 式 (J5)；不同于 sampled-token 向量 $g_t^{\mathrm{RL}}$ |
| $\lambda_j,\ \kappa_s,\ \mathcal U(\mathbf w)$ | 逐方向强度惩罚；整步激活固定费用；联合二次效用 | 式 (D2)；不等于训练全局系数 $\lambda_{\mathrm{ref/evi}}$ |
| $\widetilde w_j,\ V_k^*,\ \mathbf w_k^*$ | 扣 step fee 前的候选权重；最优开启效用；最终开关后的权重 | 式 (D4)–(D6)；$V_k^*$ 不是环境历史价值 $V_k^p$ |
| $a(u),\ M_{<t},\ V^*(h_k)$ | 前缀接受强度场；本 step 已累积的加权 score；生成前预计开启效用 | 式 (U3)、(D9)；$a(u)$ 不是动作 $a_k$，$M_{<t}$ 不是 rollout 数 $M$ |

### 0.6 损失、归一化与整体收益边界

| 符号 | 含义与类型 | 定义位置／阅读提示 |
|---|---|---|
| $\ell,\ \mathcal L$ | 单位置损失；聚合后的训练损失 | 第 3、7 节；`ref`、`evi`、`GRPO` 标识通道 |
| $\lambda_{\mathrm{ref}},\ \lambda_{\mathrm{evi}}$ | 全局 anchor 强度；全局 evidence 强度 | 式 (L2)；不同于局部接受权重 $w$ |
| $m^{\mathrm{pol}},\ m^{\mathrm{evi}}$ | 有效策略 token mask；附加 evidence 可用性与路由后的 mask | 式 (L1)；均为二值量 |
| $N,\ Z_e$ | policy／reference 与 evidence 的归一化分母 | 式 (L1)；$Z_e$ 不是配分函数 $Z(w)$ |
| $\varrho_{i,t},\ \epsilon_A,\ \epsilon$ | PPO 概率比；advantage 标准化稳定项；clipping 宽度或局部数值稳定项 | 式 (Z5)–(Z6)；无下标 $\epsilon$ 的作用依所在公式判断 |
| $\varepsilon,\ \delta$ | 方向微扰参数；正性裕量或约束预算 | 式 (T1)、第 13 节、附录 B；不是实际优化器步长 $\eta$ |
| $\operatorname{sg}[\cdot],\ [x]_+,\ \operatorname{clip}$ | 停止梯度；$\max(x,0)$；截断到指定区间 | 式 (Z11)–(Z12)；均为运算符 |
| $d_k^q,\ d_k^{\widetilde p}$ | 新策略访问第 $k$ 步历史的概率分布 | 式 (P2)–(P4)；不同于动作方向 $d_k(a)$ |
| $\epsilon_k(h),\ \operatorname{TV},\ \operatorname{span}(Q)$ | 策略实现误差上界；total variation 距离；价值最大值减最小值 | 式 (P4)；用于量化目标到实际 Student 的误差 |
| $\lambda_r,\ \lambda_e,\ a(v),\ \mu$ | 附录自由策略问题的 reference／evidence 系数、FKL 系数向量、归一化乘子 | 式 (B1)–(B3)；$a(v)$ 不表示 Agent 动作或前缀控制场 |

### 0.7 最容易混淆的几组符号

| 容易混淆的一组 | 一句话区分 |
|---|---|
| $q^0$、$q_b$、受控策略 $q$ | 无证据 Teacher、证据路径起点、新 Student 策略分别承担不同角色；看所在章节与条件变量 |
| $s$、$F(p)s$、$\zeta$ | logit 向量方向、概率向量位移、采样 token 上的标量 score |
| $\widehat b$、$b(u)$、$B_k(h)$ | 单样本代理、token 条件收益、完整 step 方向收益 |
| $\widehat A_i$、$A_k^p$、$\mathcal A_t$ | GRPO 估计优势、真实动作优势、参数方向收益 |
| $h_t$ 与 $h_k$；$V_k^*$ 与 $V_k^p$ | 前者分别是收益正部／开启效用，后者分别是交互历史／环境价值 |
| $c_t$、$C_k$、$C_{ij}$、$\mathcal C_t$ | token Fisher、step Fisher、联合 score 矩阵、参数 reward 曲率 |
| $w_t^{\mathrm{old}}$、$\widetilde w_j$、$\mathbf w_k^*$ | 原平滑权重、联合二次候选权重、扣整步费用后的最终权重；三者不是同一个解 |
| $d_t$、$d_k(a)$、$d_k^q$ | 参数更新方向、动作概率方向、新策略历史访问分布；由自变量／上标辨认 |

## 1. 从最初的问题开始：为什么不能直接模仿有证据 Teacher

设 Student 已经针对问题生成一条回答。我们保持这条回答的前缀不变，让同一个 Teacher 分别在没有证据、给定正确证据、给定错误证据时预测下一个 token。

最直接的做法，是把有证据 Teacher 当作完整目标分布，让 Student 全面靠近它。但这个更新包含两种作用：

1. Teacher 即使不看证据，也与 Student 存在差异；这部分会把 Student 拉回 Teacher 的一般行为。
2. Teacher 看了证据之后又发生变化；这部分才是当前证据额外提供的纠正信号。

若不分开，就无法独立回答“需要多强的稳定锚定”和“这份证据在这个 token 上值得接受多少”。VERPO 的第一步因此是**拆分作用**，而不是先给所有蒸馏 token 加一个权重。[S1 §5；S2 FKL]

我们先处理单个固定前缀。后面再把这个局部对象放回完整序列和 Agent 轨迹。

## 2. 只引入推导真正需要的符号

令 $u=(x,y_{<t})$ 为预测当前 token 之前的前缀，词表为 $\mathcal V$。Student logits 和分布为

$$
z_\theta(u)\in\mathbb R^{|\mathcal V|},\qquad
\pi_\theta(v\mid u)=\frac{e^{z_\theta(u,v)}}{\sum_a e^{z_\theta(u,a)}}.
\tag{N1}
$$

先按温度 1 推导。在一个局部计算点令 $p=\pi_\theta(\cdot\mid u)$；进入 Agent 收益分析后，$p$ 专指冻结的行为 Student，更新中的模型仍写作 $\pi_\theta$。两者只在更新起点相等。

在同一个 Teacher 快照和同一条目标回答前缀上定义：

$$
q^0=q_\phi(\cdot\mid u,E^0),\qquad
q^+=q_\phi(\cdot\mid u,E^+),\qquad
q^-=q_\phi(\cdot\mid u,E^-).
\tag{N2}
$$

$E^0$ 是无证据条件，$E^+$、$E^-$ 是正确／错误证据条件；$q^e$ 泛指某一有证据目标。它们是对同一目标前缀的分布回放，不是三条不同的 Teacher 生成结果。

再引入两个职责不同的对象：

- $q^{\mathrm{ref}}$：独立 reference anchor，负责稳定 Student。
- $(q_b,q_e)$：当前证据路径的 base 与 target，只负责定义证据变化。

Fixed 中通常取 $q_b=q^0,q_e=q^e$；CTR 中取 $q_b=q^-,q_e=q^+$。**CTR 的 base 是错误证据分布，不意味着 reference 也改成错误证据分布。**本文默认 anchor 为 $q^{\mathrm{ref}}=q^0$。[S1 §2；S2]

所有 Teacher 量和控制权重在一次 Student 求导中停止梯度。Teacher 可以在不同更新之间按 frozen／snapshot／EMA 规则刷新；“本次求导固定”不等于“整个训练期间永不更新”。具体调度服从注册协议。[S3；S6]

## 3. Forward KL：把一般模仿与证据纠正精确拆开

### 3.1 从合法的 Teacher 路径开始

固定 $q_b,q_e$，记

$$
\Delta=q_e-q_b,\qquad \sum_v\Delta(v)=0.
\tag{F1}
$$

考虑算术插值路径

$$
q_w=q_b+w\Delta=(1-w)q_b+wq_e,\qquad 0\le w\le1.
\tag{F2}
$$

这里 $w$ 暂时只是 Teacher 路径参数；我们还没有说明如何选择它。由于端点都是概率分布，式 (F2) 在给定区间内合法。

### 3.2 逐项展开损失

Forward KL（FKL）为

$$
\begin{aligned}
D_{\mathrm{KL}}(q_w\Vert p)
&=\sum_vq_w(v)\log q_w(v)-\sum_vq_w(v)\log p(v)\\
&=D_{\mathrm{KL}}(q_b\Vert p)
-w\sum_v\Delta(v)\log p(v)
+H(q_b)-H(q_w),
\end{aligned}
\tag{F3}
$$

其中 $H(q)=-\sum_vq(v)\log q(v)$。Teacher 和 $w$ 固定后，最后两个熵只依赖 Teacher 路径，对 Student 梯度为零。因此得到两个可分别加权的 Student 项：

$$
\ell_{\mathrm{ref}}^{\mathrm F}=D_{\mathrm{KL}}(q^{\mathrm{ref}}\Vert p),\qquad
\ell_{\mathrm{evi}}^{\mathrm F}=-w\sum_v\Delta(v)\log p(v).
\tag{F4}
$$

只有 $q^{\mathrm{ref}}=q_b$ 且两项系数匹配时，它们才直接对应式 (F3) 的同一条 KL 路径。将 anchor 与 contrast 分开是 VERPO 的目标设计，不应把任意解耦目标重新宣称成一个混合 Teacher 的 KL。

### 3.3 为什么 evidence 的下降方向就是 $\Delta$

对 softmax logits：

$$
\frac{\partial\log p(v)}{\partial z(a)}=\mathbf1\{v=a\}-p(a).
\tag{F5}
$$

所以

$$
\begin{aligned}
\nabla_z\ell_{\mathrm{ref}}^{\mathrm F}&=p-q^{\mathrm{ref}},\\
\frac{\partial\ell_{\mathrm{evi}}^{\mathrm F}}{\partial z(a)}
&=-w\left[\Delta(a)-p(a)\sum_v\Delta(v)\right]
=-w\Delta(a).
\end{aligned}
\tag{F6}
$$

梯度下降分别沿 $q^{\mathrm{ref}}-p$ 与 $w\Delta$ 移动。至此，开头的两种作用获得了明确的数学对象：前者是恢复／锚定，后者是证据条件改变带来的修正。

要注意坐标：$\Delta$ 最初是两个 Teacher 概率的差，但式 (F6) 说明它在 Student 更新中扮演 **logit 下降方向**。Student 的概率变化还需要经过 softmax Jacobian。这一步将在下一节展开。[S1 §5；S2 FKL]

## 4. Reverse KL：为什么改变 KL 方向就必须改变证据坐标

### 4.1 先得到 softmax 的局部几何

对式 (N1) 求导，得到 categorical Fisher 矩阵：

$$
F(p)=\frac{\partial p}{\partial z}
=\operatorname{diag}(p)-pp^\top.
\tag{G1}
$$

对任意向量 $a,b$，

$$
a^\top F(p)b
=\mathbb E_p[ab]-\mathbb E_p[a]\mathbb E_p[b],\qquad
F(p)\mathbf1=0.
\tag{G2}
$$

若在 logits 上沿 $s$ 做微扰，$p_\varepsilon=\operatorname{softmax}(z+\varepsilon s)$，则

$$
\left.\frac{d p_\varepsilon}{d\varepsilon}\right|_0=F(p)s.
\tag{G3}
$$

设 $\psi(\varepsilon)=\log\sum_vp(v)e^{\varepsilon s(v)}$。有 $\psi'(0)=\mathbb E_ps$、$\psi''(0)=\operatorname{Var}_p(s)$，而

$$
\begin{aligned}
D_{\mathrm{KL}}(p\Vert p_\varepsilon)
&=\psi(\varepsilon)-\varepsilon\psi'(0),\\
D_{\mathrm{KL}}(p_\varepsilon\Vert p)
&=\varepsilon\psi'(\varepsilon)-\psi(\varepsilon).
\end{aligned}
\tag{G4}
$$

两者在零点的二阶项相同：

$$
D_{\mathrm{KL}}=\frac{\varepsilon^2}{2}s^\top F(p)s+O(\varepsilon^3).
\tag{G5}
$$

这解释了 Fisher 为什么随后成为“移动代价”。它是 KL 的局部曲率，不是 reward Hessian 的恒等替代。

### 4.2 RKL 的自然路径是几何插值

Reverse KL（RKL）为 $D_{\mathrm{KL}}(p\Vert q)$。它对 $\log q$ 线性，因此应定义

$$
r=\log q_e-\log q_b,\qquad
q_w^{\mathrm R}(v)=\frac{q_b(v)e^{wr(v)}}{Z(w)},\qquad
Z(w)=\sum_vq_b(v)e^{wr(v)}.
\tag{R1}
$$

假设比较支持上的 Teacher 概率为正。取对数再代入：

$$
\begin{aligned}
\log q_w^{\mathrm R}&=\log q_b+wr-\log Z(w),\\
D_{\mathrm{KL}}(p\Vert q_w^{\mathrm R})
&=D_{\mathrm{KL}}(p\Vert q_b)-w\mathbb E_p[r]+\log Z(w).
\end{aligned}
\tag{R2}
$$

$\log Z(w)$ 在本次 Student 求导中为常数，因此解耦后的 evidence 项是

$$
\ell_{\mathrm{evi}}^{\mathrm R}=-w\mathbb E_p[r],\qquad
\ell_{\mathrm{ref}}^{\mathrm R}=D_{\mathrm{KL}}(p\Vert q^{\mathrm{ref}}).
\tag{R3}
$$

### 4.3 从标量 expectation 求出 logit 方向

由于 $\nabla_z p=F(p)$ 且 $F(p)$ 对称，

$$
\begin{aligned}
-\nabla_z\ell_{\mathrm{evi}}^{\mathrm R}&=wF(p)r,\\
\nabla_z\ell_{\mathrm{ref}}^{\mathrm R}
&=F(p)(\log p-\log q^{\mathrm{ref}}),\\
\xi(v):=[F(p)r](v)&=p(v)\left[r(v)-\mathbb E_p r\right].
\end{aligned}
\tag{R4}
$$

$\xi$ 才是 RKL evidence 的单位 logit 下降方向。$r$ 是 log-density contrast；$q_e-q_b$ 是 FKL 的方向来源，不能直接拿来替代 RKL 的 $\xi$。

现在两条路线终于能接入同一个接口：

$$
\boxed{
s=\begin{cases}
\Delta,&\mathrm{FKL},\\
F(p)r,&\mathrm{RKL}.
\end{cases}}
\qquad
\text{logit 位移为 }s,\quad \text{一阶概率位移为 }F(p)s.
\tag{R5}
$$

对于 RKL，概率位移因而是 $F(p)^2r$。这不是重复误乘 Fisher：第一次来自 RKL 损失对 logits 的梯度，第二次来自 logits 到概率的映射。[S1 §8、§11；S2 RKL]

## 5. Fixed、CTR、FEC：方向应该由哪些证据条件构造

FKL／RKL 回答了“分布差异怎样变成更新方向”。接下来才回答“应该比较哪两种证据”。这两条轴独立，Fixed、CTR、FEC 共用后续控制器。

### 5.1 Fixed：有证据与无证据的增量

$$
\Delta^{\mathrm{Fixed}}=q^e-q^0,\qquad
r^{\mathrm{Fixed}}=\log q^e-\log q^0.
\tag{E1}
$$

它度量这份证据整体造成的变化，可能同时包含内容、表达方式和 evidence prompt 的影响。“Fixed”在这里是证据模式名称，Teacher 是否 EMA 是另一条实验轴。

### 5.2 CTR：正确与错误证据对照

Contrastive evidence（CTR）比较相同提示框架中的正确、错误证据：

$$
\Delta^{\mathrm{CTR}}=q^+-q^-,\qquad
r^{\mathrm{CTR}}=\log q^+-\log q^-.
\tag{E2}
$$

多个负证据回放先在概率空间平均：

$$
q^-=\frac1K\sum_{j=1}^Kq^{-,j}.
\tag{E3}
$$

RKL 中使用 $\log q^-$，即**先平均概率，再取 log**；不是平均原始 logits 或 log 概率。正负分支应使用同一 Teacher、目标前缀与 evidence scaffold。负池排除当前目标；负池为空只屏蔽 evidence 通道，保留 reference 与 outcome 通道。正证据来源服从各协议，不能把外部参考解和 rollout 正例悄悄互换。[S1 §2；S3；S6]

### 5.3 FEC：为什么还需要第三个无证据分布

CTR 试图抵消“提供了证据”这个共同影响，但两个 evidence 条件可能并不完全对称。Fisher Evidence Contrast（FEC）引入 $q^0$，用正负中点相对无证据条件的变化估计共同成分。

理解该构造需要一个局部模型。在相应坐标中，设

$$
q^+\approx q^0+n+d_+,\qquad q^-\approx q^0+n+d_-.
\tag{E4}
$$

若 $d_++d_-\approx0$，中点减 $q^0$ 近似共同分量 $n$。如果对称性不成立，中点也会含有任务信息；因此这是一项可检验的建模假设，不是由三次回放自动获得的因果分离定理。

FKL 定义

$$
\Delta^{\mathrm{task}}=q^+-q^-,\qquad
\Delta^{\mathrm{nuis}}=\frac{q^++q^-}{2}-q^0.
\tag{E5}
$$

求 Fisher 度量下的带 ridge 一维回归：

$$
\alpha_{\mathrm F}
=\arg\min_\alpha
\left\{\frac12\|\Delta^{\mathrm{task}}-\alpha\Delta^{\mathrm{nuis}}\|_{F(p)}^2
+\frac{\epsilon_{\mathrm{fec}}}{2}\alpha^2\right\}.
\tag{E6}
$$

对 $\alpha$ 求导并令零：

$$
\begin{aligned}
\alpha_{\mathrm F}
&=\frac{(\Delta^{\mathrm{task}})^\top F(p)\Delta^{\mathrm{nuis}}}
{(\Delta^{\mathrm{nuis}})^\top F(p)\Delta^{\mathrm{nuis}}+\epsilon_{\mathrm{fec}}},\\
s^{\mathrm{F,FEC}}=\Delta^{\mathrm{FEC}}
&=\Delta^{\mathrm{task}}-\alpha_{\mathrm F}\Delta^{\mathrm{nuis}}.
\end{aligned}
\tag{E7}
$$

若 ridge 为零且 nuisance 范数非零，残差与 nuisance Fisher 正交；正 ridge 下内积等于 $\alpha_{\mathrm F}\epsilon_{\mathrm{fec}}$，不是严格零。FKL 残差仍满足坐标和为零，可直接用于 signed evidence loss；它不必是两个合法概率端点之间的插值。

### 5.4 RKL-FEC 必须先映射，再投影

RKL 在 log-density 坐标中提出对应假设，定义

$$
\begin{aligned}
r^{\mathrm{task}}&=\log q^+-\log q^-,\\
r^{\mathrm{nuis}}&=\tfrac12(\log q^++\log q^-)-\log q^0,\\
\xi^{\mathrm{task}}&=F(p)r^{\mathrm{task}},\qquad
\xi^{\mathrm{nuis}}=F(p)r^{\mathrm{nuis}}.
\end{aligned}
\tag{E8}
$$

依照原仓库合同，在实际 logit 方向 $\xi$ 的 Fisher 几何中回归：

$$
\begin{aligned}
\alpha_{\mathrm R}
&=\frac{(\xi^{\mathrm{task}})^\top F(p)\xi^{\mathrm{nuis}}}
{(\xi^{\mathrm{nuis}})^\top F(p)\xi^{\mathrm{nuis}}+\epsilon_{\mathrm{fec}}},\\
r^{\mathrm{FEC}}&=r^{\mathrm{task}}-\alpha_{\mathrm R}r^{\mathrm{nuis}},\\
s^{\mathrm{R,FEC}}&=F(p)r^{\mathrm{FEC}}.
\end{aligned}
\tag{E9}
$$

实现 evidence 项为 $-w\mathbb E_{\pi_\theta}r^{\mathrm{FEC}}$；若保留归一化路径常数，可从 $q_w\propto q^-e^{wr^{\mathrm{FEC}}}$ 的 KL 差得到相同 Student 梯度。reference 仍独立指向 $q^0$。Teacher、$\alpha$、$w$ 停梯度；Student expectation 保留梯度。[S1 §10；S2 FEC；S3]

至此，六种组合都输出一个确定的 $s$。FEC 去除的是 measured nuisance 所线性解释的分量，若 nuisance 估错会减去有用方向。**是否接受这个残差，还需要 outcome；这正是下一节 ZPD 的问题。**

## 6. 从 RL 收益出发推导原版逐 token ZPD

### 6.1 先提出理想问题，再解释可计算近似

有了方向 $s_t$，并不意味着应该全量使用。证据可能不可靠，也可能在当前 Student 已掌握的位置造成不必要的移动。ZPD（zone of proximal development）在这里用 outcome 改善与移动代价定义“可接受的纠正”。

令 $D_t=\partial z_t/\partial\theta$ 为 logit Jacobian。在普通欧氏梯度下降下，单位 evidence 参数方向是

$$
d_t=D_t^\top s_t.
\tag{Z1}
$$

把其他已固定更新吸收到局部基点 $\bar\theta$，理想问题为

$$
w_t^\star=\arg\max_{0\le w\le1}J_{\mathrm{RL}}(\bar\theta+\eta w d_t).
\tag{Z2}
$$

我们不知道更新后的真实任务收益函数。先写二阶展开：

$$
\begin{aligned}
\Delta J(w)&=\eta w\mathcal A_t-\tfrac12\eta^2w^2\mathcal C_t+O((\eta w)^3),\\
\mathcal A_t&=\nabla_\theta J_{\mathrm{RL}}^\top d_t,\qquad
\mathcal C_t=-d_t^\top\nabla_\theta^2J_{\mathrm{RL}}d_t.
\end{aligned}
\tag{Z3}
$$

仅当 $\mathcal C_t>0$，这个二次模型才具有收益递减的凹形状。此时约束解为

$$
w_t^{\mathrm{quad}}=\operatorname{clip}\left(\frac{[\mathcal A_t]_+}{\eta\mathcal C_t},0,1\right).
\tag{Z4}
$$

这一步给出的是“收益除以代价”的结构。它既未给出可计算的真实 $\mathcal A_t,\mathcal C_t$，也未证明真实 reward 函数处处凹。

### 6.2 从 GRPO 得到 sampled-token 收益代理

同一 prompt 的 $M$ 条 rollout 奖励为 $R_i$。原版 Group Relative Policy Optimization（GRPO）常用组标准化优势

$$
\widehat A_i=\frac{R_i-\bar R}{\operatorname{std}(R)+\epsilon_A},\qquad
\varrho_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid u_{i,t})}{p_{\mathrm{old}}(y_{i,t}\mid u_{i,t})}.
\tag{Z5}
$$

其 clipped surrogate 的单 token 负目标为

$$
\ell_{\mathrm{GRPO},i,t}
=-\min\left(\varrho_{i,t}\widehat A_i,
\operatorname{clip}(\varrho_{i,t},1-\epsilon,1+\epsilon)\widehat A_i\right).
\tag{Z6}
$$

在更新起点且 clipping 未关闭梯度时，reward 上升的局部 logit 方向为

$$
g_{i,t}^{\mathrm{RL}}=\widehat A_i(e_{y_{i,t}}-p_{i,t}).
\tag{Z7}
$$

保留共享参数影响时，某个 evidence 位置 $t$ 的一阶收益包含所有位置 $r$：

$$
\mathcal A_t\approx\sum_r(g_r^{\mathrm{RL}})^\top D_rD_t^\top s_t.
\tag{Z8}
$$

原版 practical ZPD 先只保留 $r=t$，再近似 $D_tD_t^\top\approx\kappa_t I$、吸收未知正尺度，得到

$$
\widehat b_t=\widehat A_i(e_{y_t}-p_t)^\top s_t
=\widehat A_i\left[s_t(y_t)-\mathbb E_{v\sim p_t}s_t(v)\right].
\tag{Z9}
$$

因此 $\widehat b_t$ 是 sampled-token outcome alignment。它不是 Teacher 置信度，也不是精确的参数更新收益。这里已经使用了局部 RL surrogate、忽略跨位置 Jacobian、局部各向同性等近似。[S1 §7、§11、§19]

### 6.3 移动代价为何写成 Fisher 方差

真实的 reward 曲率可能不定且昂贵。原版用正半定的局部 KL 曲率作移动代理。由式 (G5)，定义

$$
c_t=s_t^\top F(p_t)s_t
=\mathbb E_{p_t}[s_t^2]-\mathbb E_{p_t}[s_t]^2\ge0.
\tag{Z10}
$$

它不要求建立词表大小的 Fisher 矩阵，只需计算一阶、二阶矩。为了数值稳定，原稿再引入 $c_0,\beta,\rho$ 等尺度与 floor。本文用两个有效参数统一写成

$$
h_t=[\widehat b_t]_+,\qquad
d_t^{\mathrm{cost}}=a_c c_t+\epsilon_c,\qquad a_c\ge0,\ \epsilon_c>0.
\tag{Z11}
$$

这里 $d_t^{\mathrm{cost}}$ 是标量分母，区别于式 (Z1) 的参数向量 $d_t$。它与旧写法的关系是 $a_c=\tau\beta$、$\epsilon_c=\tau(c_0+\rho)$。

### 6.4 原版平滑权重是怎样选择的

沿用收益／代价结构，硬截断代理是 $\operatorname{clip}(h_t/d_t^{\mathrm{cost}},0,1)$；仓库 canonical controller 采用平滑形式

$$
\boxed{w_t^{\mathrm{old}}=\operatorname{sg}\left(\frac{h_t}{h_t+a_c c_t+\epsilon_c}\right).}
\tag{Z12}
$$

它在 $h_t=0$ 时关闭，收益增加时增大，代价增加时减小，并保持 $0\le w_t<1$。`compute_verpo_token_weights` 当前使用 `cost_alpha`、`cost_epsilon`，分别对应 $a_c,\epsilon_c$。[S3]

**式 (Z12) 是平滑控制器设计，不能从式 (Z3) 不加额外假设地声称“精确求解得到”。**为说明差别，可以反向构造一个具有该解的设计目标：

$$
\mathcal U_{\mathrm{smooth}}(w)=h_tw-\tfrac12(h_t+d_t^{\mathrm{cost}})w^2.
\tag{Z13}
$$

求导得 $h_t-(h_t+d_t^{\mathrm{cost}})w=0$，确实给出式 (Z12)。但式 (Z13) 的曲率显式含收益 $h_t$；它是本文用于解释权重形状的代数重表达，不是原真实 reward Hessian 或后文精确 KL 目标。这样就能把原版的工程公式与它的理论动机串起来，同时保留两者边界。

## 7. 原版 VERPO 的完整训练目标与控制层级

前面已经分别说明 outcome、reference、evidence 的来源，现在才组合最终目标。

令 $m^{\mathrm{pol}}_{i,t}$ 表示有效策略 token，$m^{\mathrm{evi}}_{i,t}$ 还包含 evidence 可用性、可选 group gate 与 rollout scope。定义

$$
N=\max\left(1,\sum_{i,t}m^{\mathrm{pol}}_{i,t}\right),\qquad
Z_e=\max\left(1,\sum_{i,t}m^{\mathrm{evi}}_{i,t}\right).
\tag{L1}
$$

为便于展示，以下采用 token 均值；实际 protocol 若采用 sequence mean，必须显式替换 reduction，不能把不同长度聚合说成相同目标。

$$
\boxed{
\mathcal L_{\mathrm{VERPO}}=
\mathcal L_{\mathrm{GRPO}}+\lambda_{\mathrm{ref}}\mathcal L_{\mathrm{ref}}
+\lambda_{\mathrm{evi}}\mathcal L_{\mathrm{evi}}.
}
\tag{L2}
$$

FKL 的两项为

$$
\begin{aligned}
\mathcal L_{\mathrm{ref}}^{\mathrm F}
&=\frac1N\sum_{i,t}m^{\mathrm{pol}}_{i,t}D_{\mathrm{KL}}(q^0_{i,t}\Vert\pi_{\theta,i,t}),\\
\mathcal L_{\mathrm{evi}}^{\mathrm F}
&=-\frac1{Z_e}\sum_{i,t}m^{\mathrm{evi}}_{i,t}\operatorname{sg}[w_{i,t}]
\sum_v\operatorname{sg}[\Delta_{i,t}(v)]\log\pi_{\theta,i,t}(v).
\end{aligned}
\tag{L3}
$$

RKL 则为

$$
\begin{aligned}
\mathcal L_{\mathrm{ref}}^{\mathrm R}
&=\frac1N\sum_{i,t}m^{\mathrm{pol}}_{i,t}D_{\mathrm{KL}}(\pi_{\theta,i,t}\Vert q^0_{i,t}),\\
\mathcal L_{\mathrm{evi}}^{\mathrm R}
&=-\frac1{Z_e}\sum_{i,t}m^{\mathrm{evi}}_{i,t}\operatorname{sg}[w_{i,t}]
\mathbb E_{v\sim\pi_{\theta,i,t}}\operatorname{sg}[r_{i,t}(v)].
\end{aligned}
\tag{L4}
$$

$\Delta,r$ 分别按 Fixed／CTR／FEC 构造。完整训练目标中的全局系数、mask 与归一化会影响真实更新强度，不能把 $w$ 单独解释成实际 KL 位移。

| 控制量 | 职责 | 不能替代什么 |
|---|---|---|
| $\lambda_{\mathrm{ref}}$ | 全局 reference 锚定 | 不是 evidence 是否有益的判断 |
| $\lambda_{\mathrm{evi}}$ | 全局 evidence 强度 | 不能替代逐位置收益估计 |
| group／availability／scope mask | 决定哪些 evidence 样本可用 | 不应关闭独立 reference 或 GRPO |
| $w_t$ | 接受多少当前方向 | 不负责改变 Fixed／CTR／FEC 的定义 |
| 学习率与优化器 | 把梯度转成参数变化 | 不等于 policy-space 的理想步长 |

历史 V26 曾使用 mixed group 门控；当前默认 `group_zpd_enabled=false`。这只关闭显式 group 筛选，不消除 token ZPD：若一组回报完全相同，$\widehat A_i=0$ 仍会令式 (Z9)–(Z12) 的权重为零。方向可计算、样本可用、收益可识别是三件不同的事。[S3；S6]

## 8. 原版已经回答了什么，为什么还需要 Agent 推导

到这里，原 VERPO 的完整逻辑已经闭合：**Teacher 条件比较产生方向，FEC 处理所估计的共同成分，sampled outcome alignment 与 Fisher 决定权重，加性目标执行更新。**

但多步 Agent 还提出三个原式没有直接解决的问题。

第一，一个工具调用的价值不能只由当前文本判断。它可能暂时失败，却产生对后续解决问题有用的 observation。应衡量“做完这一步之后，普通 Student 继续执行的回报”。

第二，同一步中可能同时存在有益和有害 token 方向。若先按整步总分关掉整个 step，可能错过有益子集；反之，逐 token 各算一个 gate 再乘整步 gate，也不自动是同一目标的最优解。

第三，式 (Z9) 的一条样本值，不等于该前缀的真实条件收益。同一个 prompt 的 8 条轨迹通常在第一步之后就分岔。需要区分总体方向收益、有限样本估计和据此作出的非线性选择。

因此，Agent 扩展保留第 3–5 节构造的方向 $s$，把“它的价值是什么”重新放到真实交互过程里定义。下面从环境和历史开始，而不直接跳到联合最优权重。

## 9. 把单段回答放进多步 Agent 交互

有限时域内，第 $k$ 步之前的可见历史为

$$
h_k=(x,a_1,o_1,\ldots,a_{k-1},o_{k-1}),\qquad
u_{k,t}=(h_k,y_{k,<t}).
\tag{A1}
$$

$a_k=(y_{k,1},\ldots,y_{k,T_k})$ 是完整策略输出，$o_k$ 是工具或环境返回。由冻结行为策略 $p$ 生成动作：

$$
p_k(a_k\mid h_k)=\prod_{t=1}^{T_k}p(y_{k,t}\mid u_{k,t}).
\tag{A2}
$$

reasoning、工具名、参数、最终回答和 EOS 只要由策略生成且影响过程，都计入动作。observation 保留在后续上下文，但不进入策略概率乘积，也不作为策略输出 loss；padding 分数为零。

令 $r_k$ 为环境奖励，$G_k=\sum_{j=k}^Hr_j$，定义

$$
\begin{aligned}
Q_k^p(h,a)&=\mathbb E_p[G_k\mid h_k=h,a_k=a],\\
V_k^p(h)&=\mathbb E_{a\sim p_k(\cdot\mid h)}Q_k^p(h,a),\\
A_k^p(h,a)&=Q_k^p(h,a)-V_k^p(h),\\
J(p)&=\mathbb E_p\sum_{k=1}^Hr_k.
\end{aligned}
\tag{A3}
$$

$Q_k^p$ 指当前动作之后由**旧 Student 继续**，包括新工具结果及后续决策；不把 Teacher 接管后的成绩当成 Student 的价值。单段回答是 $H=1$ 的特例，纯终局奖励时各尚未结束 step 的 return-to-go 都等于终局 $R$。

后续恒等式采用以下条件：有限时域与动作长度、固定共同支持、有界回报；本批次冻结 $p$、Teacher 和环境机制；$s(u,v)$ 在当前 token 产生前由前缀与固定证据确定；所定义的新策略对每个相关前缀都可执行。

**因果 Teacher forcing 只保证序列内部不看未来，不自动保证 evidence prompt 没有使用未来。**用当前目标轨迹的未来 observation、最终奖励或事后挑选的 sibling 构造方向，会破坏所需独立性。这样的训练仍可作为 R0 经验算法研究，但不能直接继承下文的可预测方向定理。

## 10. 单 token 收益：为什么原来的 alignment 会再次出现

### 10.1 先求概率路径的导数

在一个固定前缀 $u$ 上，使用第 3–5 节给出的 logit 方向 $s(u,v)$，定义

$$
p_\varepsilon(v\mid u)=\frac{p(v\mid u)e^{\varepsilon s(u,v)}}{\sum_ap(a\mid u)e^{\varepsilon s(u,a)}}.
\tag{T1}
$$

对数为 $\log p+\varepsilon s-\psi_u(\varepsilon)$。求导得到

$$
\zeta(u,y)
:=\left.\partial_\varepsilon\log p_\varepsilon(y\mid u)\right|_0
=s(u,y)-\mathbb E_{v\sim p(\cdot\mid u)}s(u,v).
\tag{T2}
$$

因此 $\mathbb E_p[\zeta\mid u]=0$。它与原式 (Z9) 的中心化方向完全相同，区别在于现在明确了行为分布和可预测性条件。

### 10.2 再对后续回报求导

定义 token 继续价值 $Q^p(u,v)=\mathbb E_p[G_k\mid u,y=v]$。只改变当前条件分布、之后恢复旧 Student，则

$$
\begin{aligned}
b(u)
&:=\left.\partial_\varepsilon\sum_vp_\varepsilon(v\mid u)Q^p(u,v)\right|_0\\
&=\sum_vp(v\mid u)Q^p(u,v)\zeta(u,v)\\
&=\mathbb E_p[G_k\zeta\mid u]
=\mathbb E_p[(G_k-\beta(u))\zeta\mid u].
\end{aligned}
\tag{T3}
$$

最后一步使用条件零均值；$\beta(u)$ 应在当前采样前固定，或满足相应条件独立性。

**原版用 $\widehat A_i\zeta$ 作代理，Agent 理论用 $\mathbb E[(G_k-\beta)\zeta\mid u]$ 定义真实方向收益。**这就是两份推导的连接点。原版组标准化 advantage、长度 shaped reward 与原始任务 return 的量纲和 estimand 不同，不能只换名称就称为同一个 $b(u)$。

$b(u)>0$ 表示沿这份固定方向存在正的一阶局部收益；一条样本 $(G_k-\beta)\zeta>0$ 只提供一个有噪声的观测，不能判定条件均值必为正。

### 10.3 Fisher 仍然是同一个局部代价

由式 (G5)：

$$
c(u)=\mathbb E_p[\zeta^2\mid u]=s(u)^\top F(p_u)s(u).
\tag{T4}
$$

当 $c=0$ 时，$s$ 在行为支持上是常数，故 $\zeta=0$、$b=0$。一个预测器若同时给出接近零的 $c$ 和很大的 $b$，应作为不一致信号处理。

若条件回报在 $[0,1]$，Cauchy–Schwarz 还给出

$$
|b(u)|^2\le\operatorname{Var}(G_k\mid u)c(u)\le\tfrac14c(u).
\tag{T5}
$$

这是总体条件量的约束；有噪声的有限样本值越界，不自动等于代码错误。一般回报范围 $[r_{\min},r_{\max}]$ 将 $1/4$ 替换为 $(r_{\max}-r_{\min})^2/4$。

## 11. 从 token 到完整 step：收益可加，Fisher 为什么也可加

固定 $h_k=h$，在这一完整动作的各策略 token 上沿 $s$ 同时作小强度扰动。由对数概率可加：

$$
S_k(a)=\left.\partial_\varepsilon\log p_{k,\varepsilon}(a\mid h)\right|_0
=\sum_t\zeta_{k,t},\qquad
d_k(a)=p_k(a\mid h)S_k(a).
\tag{S1}
$$

由于各条件 score 均值为零，$\mathbb E_p[S_k\mid h]=0$，即 $\sum_a d_k(a)=0$。$d_k$ 才是完整动作概率空间的方向。

只改变当前完整动作的分布，旧 continuation 不动，其收益为

$$
\begin{aligned}
B_k(h)
&=\sum_ad_k(a)Q_k^p(h,a)\\
&=\mathbb E_p[G_kS_k\mid h]
=\mathbb E_p[A_k^p(h,a)S_k\mid h]\\
&=\mathbb E_p\left[\sum_tb(u_{k,t})\mid h\right].
\end{aligned}
\tag{S2}
$$

最后一行来自全期望公式。它说明 step 收益由访问到的 token 条件收益累积而来，但不是把每个 token 的单样本值都视为已知真值。

对 $t<r$，$\zeta_t$ 在生成第 $r$ 个 token 前已知，所以

$$
\mathbb E[\zeta_t\zeta_r\mid h]
=\mathbb E[\zeta_t\mathbb E[\zeta_r\mid u_r]\mid h]=0.
\tag{S3}
$$

展开平方即可得

$$
\boxed{
C_k(h)=\mathbb E_p[S_k^2\mid h]
=\mathbb E_p\left[\sum_tc(u_{k,t})\mid h\right].
}
\tag{S4}
$$

这是期望中的 Fisher chain rule，单条轨迹并不满足 $(\sum_t\zeta_t)^2=\sum_tc_t$。它也不意味着共享参数训练中各 token 不相互影响；式 (Z8) 的 Jacobian 交叉项仍然存在。

一个完整 step 的概率按 token 概率相乘，所以对应 score 按 token **求和**。若除以每条长度，会改变优化对象。可以设计长度归一化 surrogate，但需另行说明它与上述动作分布目标的区别。

## 12. 先求整步标量控制，理解它为什么还不够

### 12.1 从局部切向量构造一条有限的合法路径

式 (S1) 给出了零点方向 $pS$。为了分析有限强度的精确 KL，可以另定义完整动作的 affine 路径：

$$
q_w(a\mid h)=p(a\mid h)[1+wS(a)],\qquad 0\le w\le\bar w.
\tag{C1}
$$

由于 $\mathbb E_pS=0$，它归一化；还必须选择 $\bar w$，保证所有合法动作上的 $1+wS(a)>0$。这里 $q_w$ 表示受控 Student 动作分布，与前面 Teacher 路径 $q_w^{\mathrm F},q_w^{\mathrm R}$ 不是同一对象。

式 (C1) 与逐 token softmax 微扰只在零点具有相同切向量；有限强度时，两条路径通常不同。选 affine 路径的原因是它允许精确写出收益和 KL，而不是声称实际模型天然按这条路径更新。

### 12.2 精确收益与精确 KL

在当前历史且旧 continuation 固定时，

$$
\sum_a[q_w(a)-p(a)]Q^p(h,a)=wB(h).
\tag{C2}
$$

记 $K(w)=D_{\mathrm{KL}}(q_w\Vert p)$，利用 $\mathbb E_p[wS]=0$：

$$
\begin{aligned}
K(w)&=\mathbb E_p[(1+wS)\log(1+wS)-wS],\\
K'(w)&=\mathbb E_p[S\log(1+wS)],\\
K''(w)&=\mathbb E_p\frac{S^2}{1+wS}\ge0,\qquad K'(0)=0.
\end{aligned}
\tag{C3}
$$

于是有一个明确的标量优化问题：

$$
\max_{0\le w\le\bar w}\; wB-\tau K(w),\qquad \tau>0.
\tag{C4}
$$

目标为凹函数。若方向非退化，$B\le0$ 时取零最优；$B>0$ 时零点导数为正，存在有益的足够小正步长。内部最优点满足 $B=\tau K'(w)$，若到达上界仍在增大则取上界。

在零点 $K''(0)=C$，因此局部二次模型给出

$$
w^{\mathrm{local}}=\operatorname{clip}\left(\frac{B}{\tau C},0,\bar w\right),\qquad C>0.
\tag{C5}
$$

这一步终于从明确的 policy-space KL 目标得到了 $B/(\tau C)$。它与式 (Z4) 的参数空间 reward Taylor 模型结构相近，但假设、曲率来源和实现对象均不同；也不等于原平滑式 (Z12)。

### 12.3 标量控制的限制来自“所有位置绑在一起”

式 (C4) 只允许所有 token 使用共同强度。如果一部分方向收益为正，另一部分为负，总 $B$ 可能为负，导致整步被拒绝。我们希望能保留正收益子集，所以应把控制变量从一个 $w$ 扩展为一组 $\mathbf w$。这就是联合优化的动机。

## 13. 精确 Step–Token 联合优化：把位置选择写进同一个目标

### 13.1 先定义可预测的方向基底

固定历史 $h$，设 $f_j(u)$ 为预先定义的 $d$ 个基底，例如“第 $j$ 个生成位置”或由当前前缀可判断的 token 类别。令

$$
S_j(a)=\sum_tf_j(u_t)\zeta_t,\qquad
B_j=\mathbb E_p[G_kS_j\mid h],\qquad
C_{ij}=\mathbb E_p[S_iS_j\mid h].
\tag{J1}
$$

“最终出错的 token”是事后标签，不能直接当成可预测基底。联合理论需要定义所有候选前缀的控制场，而不仅是当前采到的一串位置。

令 $\mathbf S=(S_1,\ldots,S_d)^\top$，推广 affine 动作分布为

$$
q_{\mathbf w}(a\mid h)=p(a\mid h)(1+\mathbf w^\top\mathbf S(a)),\qquad
\mathbf w\in\mathcal W.
\tag{J2}
$$

$\mathcal W$ 可包含非负性、盒约束以及全支持正性条件 $1+\mathbf w^\top\mathbf S(a)>0$。严格理论可用 $\ge\delta>0$ 的紧致可行域确保最大值存在。

### 13.2 直接推导联合目标、梯度与曲率

与标量情形相同，收益精确为 $\mathbf w^\top\mathbf B$。设 $x=\mathbf w^\top\mathbf S$，则

$$
\begin{aligned}
K(\mathbf w)&=\mathbb E_p[(1+x)\log(1+x)-x],\\
\nabla K(\mathbf w)&=\mathbb E_p[\mathbf S\log(1+x)],\\
\nabla^2K(\mathbf w)&=\mathbb E_p\left[\frac{\mathbf S\mathbf S^\top}{1+x}\right]\succeq0.
\end{aligned}
\tag{J3}
$$

得到

$$
\boxed{\max_{\mathbf w\in\mathcal W}\;\mathbf w^\top\mathbf B-\tau K(\mathbf w).}
\tag{J4}
$$

在凸可行域上，这是凹最大化。若方向线性相关或曲率退化，不能保证权重唯一。

若盒约束本身已满足全部正性条件，定义 $g_j=B_j-\tau\partial_jK$，则 Karush–Kuhn–Tucker（KKT）条件为

$$
\begin{cases}
g_j\le0,&w_j=0,\\
g_j=0,&0<w_j<\bar w_j,\\
g_j\ge0,&w_j=\bar w_j.
\end{cases}
\tag{J5}
$$

当另有耦合正性或预算约束时，需加入对应乘子，不能继续把每个坐标当作独立问题。式 (J5) 使“选哪些位置”与“各选多大”成为同一最优性条件。

### 13.3 为什么零点可以对角，有限强度仍会耦合

由式 (S3) 的条件零均值，

$$
C_{ij}=\mathbb E_p\left[\sum_tf_i(u_t)f_j(u_t)c(u_t)\mid h\right].
\tag{J6}
$$

若基底作用于互不重叠的 token 位置，$f_if_j=0$，则 $i\ne j$ 时 $C_{ij}=0$。但非零权重下式 (J3) 的分母依赖所有方向，Hessian 通常重新出现交叉项。

所以：**零点的 token Fisher 正交支持小步长对角近似，不支持“任意强度联合更新互不干扰”。**若多个 Teacher 方向作用于同一个 token、基底重叠或研究共享参数更新，零点也可能非对角。

在一般联合问题中，某个 $B_j\le0$ 的坐标仍可能通过降低整体联合代价而被选中。简单“负收益坐标一律置零”的规则只适用于后面的非负、对角二次模型，不能上升为式 (J4) 的一般定理。

## 14. 可实现的对角解：先选择 token，再判断整步是否值得开启

### 14.1 明确近似和新增设计项

在零点附近，

$$
K(\mathbf w)=\tfrac12\mathbf w^\top C\mathbf w+O(\|\mathbf w\|^3).
\tag{D1}
$$

当 $C$ 对角时，加入可选强度惩罚 $\lambda_j\ge0$、整步激活费用 $\kappa_s\ge0$，得到

$$
\mathcal U(\mathbf w)=
\sum_j\left[(B_j-\lambda_j)w_j-\frac\tau2C_jw_j^2\right]
-\kappa_s\mathbf1\{\mathbf w\ne0\},\quad 0\le w_j\le\bar w_j.
\tag{D2}
$$

$\lambda_j$ 与 $\kappa_s$ 是新增的资源／稀疏设计，不是 KL 恒等式自动产生的量。首版可令它们为零；若 Teacher 已完成全部 replay，事后关闭 step 不能省掉已经付出的 Teacher 成本。

### 14.2 对每个坐标求条件最优解

暂时不扣整步固定费用。对第 $j$ 项求导：

$$
\partial_{w_j}\mathcal U_{\mathrm{pre-fee}}=B_j-\lambda_j-\tau C_jw_j.
\tag{D3}
$$

因 $C_j>0$，其约束最大值在

$$
\widetilde w_j=\operatorname{clip}\left(\frac{B_j-\lambda_j}{\tau C_j},0,\bar w_j\right).
\tag{D4}
$$

计算最优开启效用

$$
V_k^*=\sum_j\left[(B_j-\lambda_j)\widetilde w_j-\frac\tau2C_j\widetilde w_j^2\right].
\tag{D5}
$$

“不开启任何位置”的目标是零，所以最终为

$$
\boxed{
\mathbf w_k^*=\begin{cases}
\widetilde{\mathbf w}_k,&V_k^*>\kappa_s,\\
\mathbf0,&V_k^*\le\kappa_s.
\end{cases}}
\tag{D6}
$$

这不是先算两个不相关的 gate 再相乘，而是比较“关闭”与“最优开启”两个解。等号时约定关闭。

若有 $C_j=0$，真实固定方向对应的 $B_j$ 也为零；参考内核对该坐标保守关闭。加入 $C_j+\epsilon$ 相当于增加二次 ridge，改变目标，应在配置和推导中明确。

### 14.3 一个完整可实现的小例子

设两个独立 score $\zeta_1,\zeta_2$ 各以一半概率取 $\pm1$，令回报

$$
R=0.5+0.2\zeta_1-0.3\zeta_2\in[0,1].
\tag{D7}
$$

由独立性和零均值得 $B_1=\mathbb E[R\zeta_1]=0.2$、$B_2=-0.3$、$C_1=C_2=1$。

若整步共用一个标量权重，则 $B=-0.1$，标量目标拒绝该步。若取 $\tau=1,\lambda_j=0$，联合解为

$$
\widetilde{\mathbf w}=(0.2,0),\qquad V_k^*=0.02.
\tag{D8}
$$

当 $\kappa_s=0.01$ 时，扣费后效用仍为 $0.01$，应保留第一个 token。这个例子说明联合选择的价值，但数值是有限模型代理收益，不是 LLM 实验成绩。[S5 数值内核]

## 15. 三类权重的关系：哪些继承了，哪些真的改变了

| 控制器 | 所优化／采用的对象 | 权重 | 证据等级 |
|---|---|---|---|
| 原 VERPO 平滑 ZPD | sampled benefit 与 Fisher 的平滑接受设计 | $h/(h+a_cc+\epsilon_c)$ | 当前实现；不是精确 KL 最优解 |
| Agent 标量／联合精确控制 | affine 完整动作路径上的收益减 KL | 解 $B=\tau K'$ 或联合 KKT | 指定策略族和假设下的精确目标 |
| Agent 对角二次控制 | 精确目标的局部对角近似，加可选资源项 | $\operatorname{clip}((B_j-\lambda_j)/(\tau C_j),0,\bar w_j)$，再扣 step fee | 对该 surrogate 的闭式解 |

继承的是 Teacher 证据构造、logit 方向接口、中心化 score、Fisher 移动思想和 reference／evidence 分离。新增的是交互条件价值、完整动作概率方向，以及把 step 与 token 选择写成联合问题。

还必须区分**决策时机**。式 (D6) 若用本 step 生成完成后所有 token 的统计量，只是事后训练路由。若要在生成前决定是否调用 Teacher，必须有只依赖 $h_k$ 的预测器，例如估计

$$
V^*(h_k)=\mathbb E_{a\sim p(\cdot\mid h_k)}
\sum_t\left[(b(u_t)-\lambda_t)\widetilde w(u_t)-\tfrac\tau2c(u_t)\widetilde w(u_t)^2\right].
\tag{D9}
$$

一个事后总分不能反过来当成生成前可获得的控制量。即使每个 token 的权重可由当前前缀预测，若事后再用整步效用统一开关所有早期 token，最终 gate 仍然依赖未来；在线版本应使用预先冻结的 step 预测器，或采用不回改早期决策的前缀控制。

## 16. 一个 prompt 的 8 条 rollout：估计器、样本依赖与选择偏差

### 16.1 从独立轨迹构造留一基线

假设同一 prompt 有 $M=8$ 条 iid Student 轨迹，方向由满足独立性要求的数据固定。用原始回报尺度的 leave-one-out（LOO）优势：

$$
A_i^{\mathrm{LOO}}=R_i-\frac1{M-1}\sum_{j\ne i}R_j.
\tag{I1}
$$

相比包含自身的组均值，它避免 baseline 与自身 return 的直接依赖。对某个零均值方向 score $S_i$，

$$
\begin{aligned}
\mathbb E[A_i^{\mathrm{LOO}}S_i]&=\mathbb E[R_iS_i],\\
\mathbb E[(R_i-\bar R)S_i]&=\frac{M-1}{M}\mathbb E[R_iS_i].
\end{aligned}
\tag{I2}
$$

证明：其他轨迹回报与 $S_i$ 独立，交叉期望为 $\mathbb E[R_j]\mathbb E[S_i]=0$；包含自身时减去了 $\mathbb E[R_iS_i]/M$。LOO 不修复 hindsight 证据、共享随机性或采样依赖，仍需逐项检查假设。

### 16.2 同一个初始历史可以怎样估计

所有 rollout 第一步都从同一个 $x$ 出发，因此

$$
\begin{aligned}
\widehat B_1(x)&=\frac1M\sum_iA_i^{\mathrm{LOO}}S_{i,1},\\
\widehat C_{1,\mathrm{token}}&=\frac1M\sum_i\sum_tc_{i,1,t},\\
\widehat C_{1,\mathrm{score}}&=\frac1M\sum_iS_{i,1}^2.
\end{aligned}
\tag{I3}
$$

后两个量在总体中相等，有限批次可以不同；这种差异可用于一致性诊断。LOO 各项共享其他轨迹回报，不能把它们当成独立样本直接套朴素标准误。

对于一个中间 token，可记录

$$
X_{i,k,t}=A_i^{\mathrm{LOO}}\zeta_{i,k,t},\qquad Y_{i,k,t}=c(u_{i,k,t}).
\tag{I4}
$$

纯终局奖励下式 (I4) 可用同一个 $R_i$；稠密奖励时应使用合适 return-to-go 与合法基线，不能无条件复制整条总回报。

### 16.3 为什么后续步骤不能继续声称有 8 个同历史样本

第 2 步通常满足 $h_{i,2}\ne h_{j,2}$：动作、工具返回或推理内容已经不同。把“八条轨迹的第 2 步”简单平均，只能描述访问历史混合分布，不能估计某一个具体历史的条件 $B_2(h)$。

R0 可以直接用 $X$ 作经验训练权重的输入，但应称为单样本 hindsight gate。R1 则拟合

$$
b_\phi(f(u))\approx\mathbb E[X\mid f(u)],
\tag{I5}
$$

其中 $f(u)$ 只使用当前 token 之前的特征。特征压缩后预测的是同特征类的均值，不一定等于完整历史的精确 $b(u)$。由于 $c(u)$ 可由 replay 直接计算，通常不必额外训练 cost head；若要在 Teacher replay 前筛选，才需预测相关代价。

### 16.4 估计完再在同样本上挑选，会放大噪声

即使真实 $b=0$，噪声估计 $\widehat b$ 也会时正时负。只留下 $\widehat b>0$ 的项，再用这批样本报告收益，会产生选择乐观偏差。正部、排序、除以小成本、整步阈值都可能强化这种效应。

更直接地，若某个零均值 score $\zeta\in\{-1,+1\}$，事后 gate $w=\mathbf1\{\zeta=1\}$，则

$$
\mathbb E\zeta=0,\qquad \mathbb E[w\zeta]=\tfrac12\ne0.
\tag{I6}
$$

因此不能把用当前采样 token 选出的权重当作采样前固定的合法方向场，再继续沿用原零均值证明。

R1 应按 **prompt／任务实例** 分 fold，把同一 prompt 的全部 sibling 放在同一 fold；在其他 folds 拟合，在 heldout fold 出权重。方向构造和 evidence 池也要满足需要的独立性。独立干预数据用于评估选择后的收益，不能只看训练内正 alignment 占比。

### 16.5 不必先知道每个前缀的精确价值，才能学习全局控制

对于任何已经固定、可预测的控制场 $a(u)$，对完整交互路径求 likelihood-ratio 导数：环境转移项不含策略参数，过去奖励可由条件零均值消去，于是

$$
\left.\frac{d}{d\varepsilon}J(p^{\varepsilon,a})\right|_0
=\mathbb E_p\sum_{k,t}G_k a(u_{k,t})\zeta_{k,t}.
\tag{I7}
$$

不同历史的 rollout 可以一起估计这个**全局旧策略访问分布下的方向目标**。这给规模化学习控制器提供了依据，同时不把它误称为“每个中间历史的局部收益已被精确知道”。

## 17. 怎样把控制结果真正更新到 Student

### 17.1 优先保留原 VERPO 的 signed evidence loss

如果目标是控制既有 evidence 方向，最直接的接法是把原 $w_t^{\mathrm{old}}$ 替换为明确标记的 Agent 控制权重，并保持式 (L3)／(L4) 的方向构造与通道分离。两种权重是替代实验，不默认相乘。

对于 FKL，在独立 logits 坐标中，固定 $w,\Delta$ 后负梯度严格为 $w\Delta$。对于 RKL，在冻结分析点 $p$ 有负梯度 $wF(p)r$；实际 Student 更新到 $\pi_\theta\ne p$ 后，负梯度变为 $wF(\pi_\theta)r$。这也是更新多轮后应重新回放或限制偏离的原因。

**如果 $A_i$ 已用于形成 $X=A_i\zeta$ 和控制权重，不应再额外用同一个 $A_i$ 乘 evidence 方向。**额外乘法会变成另一种 advantage modulation 目标，可能在负优势轨迹上反转方向。既有独立 GRPO 分支继续存在不属于“重复相乘”。

### 17.2 显式目标分布蒸馏可用，但方向意义会改变

另一种接法是构造合法的 token 目标

$$
q^{\mathrm{soft}}(v\mid u)=\frac{p(v\mid u)e^{w(u)s(u,v)}}{\sum_ap(a\mid u)e^{w(u)s(u,a)}}
\tag{U1}
$$

并最小化 $D_{\mathrm{KL}}(q^{\mathrm{soft}}\Vert\pi_\theta)$。这个目标与原方向在分布路径的零点相切，但在 $\pi_\theta=p$ 处，普通 KL 梯度满足

$$
-\nabla_zD_{\mathrm{KL}}(q^{\mathrm{soft}}\Vert p)
=q^{\mathrm{soft}}-p=wF(p)s+O(w^2),
\tag{U2}
$$

而不是 $ws$。所以“拟合一个倾斜目标分布”和“做一步原 signed loss 梯度下降”不是同一个有限更新。若拟合充分，前者可以逼近目标；有限 SGD／Adam 步数下应测量实际实现误差。

### 17.3 affine 完整动作目标的 token 分解

若要精确实现式 (J2)，令可预测强度场 $a(u)=\sum_jw_jf_j(u)$，前缀累积 score 为

$$
M_{<t}=\sum_{r<t}a(u_r)\zeta_r.
\tag{U3}
$$

对完整动作密度 $1+\sum_ra(u_r)\zeta_r$ 条件于当前前缀，未来 score 的条件均值为零，所以前缀密度比为 $1+M_{<t}$。再对增加一个 token 后的前缀取概率比，得到

$$
q^{\mathrm{aff}}(v\mid u_t)=p(v\mid u_t)
\frac{1+M_{<t}+a(u_t)\zeta(u_t,v)}{1+M_{<t}}.
\tag{U4}
$$

它的分子期望等于分母，故归一化；相邻 token 的密度比相乘会望远镜消去中间项，恢复完整动作式 (J2)。$M$ 在每个 Agent step 开始时重置。

这要求所有可访问前缀上分子、分母都为正。只检查 8 条样本路径不能认证全支持。若 $|s|\le L$、每步最多 $T_{\max}$ 个 token、$0\le a\le a_{\max}$，则 $2T_{\max}La_{\max}<1$ 是一个保守充分条件。

### 17.4 共享参数带来的实际移动必须单独审计

对于普通梯度步，简化忽略其他通道后有

$$
\delta\theta=\eta\sum_tw_tD_t^\top s_t,\qquad
\delta z_r\approx\eta\sum_tw_tD_rD_t^\top s_t.
\tag{U5}
$$

所以一个位置 $w_r=0$，仍可能被其他位置的训练更新影响。Adam 预条件、梯度裁剪、reference／GRPO 叠加、loss normalization 又会改变这个映射。

必须分别记录目标方向、实际 $\delta z$、实际 KL 和 off-route drift。方向余弦对齐也不充分：幅度过大仍可能破坏性能。若需要额外 old-Student trust penalty，应作为明确的目标设计；它与 Teacher reference anchor 职责不同，不能悄悄互换或重复添加。

## 18. 从局部收益到整体 Agent 回报：证明到哪里，误差在哪里

### 18.1 展开并证明 performance-difference 恒等式

设 $q$ 是同一环境中的新历史策略，$d_k^q$ 是它访问第 $k$ 步历史的分布。由 Bellman 关系，沿 $q$ 轨迹有

$$
\mathbb E_q[A_k^p(h_k,a_k)]
=\mathbb E_q[r_k+V_{k+1}^p(h_{k+1})-V_k^p(h_k)].
\tag{P1}
$$

对 $k=1,\ldots,H$ 求和，中间价值项相消；终止价值为零，初始历史分布相同，因此

$$
\boxed{
J(q)-J(p)=\sum_{k=1}^H\mathbb E_{h\sim d_k^q}
\sum_aq_k(a\mid h)A_k^p(h,a).
}
\tag{P2}
$$

由于 $\sum_ap_k(a\mid h)A_k^p(h,a)=0$，对 affine 控制策略可以代入

$$
J(q)-J(p)=\sum_k\mathbb E_{h\sim d_k^q}[\mathbf w_k(h)^\top\mathbf B_k(h)].
\tag{P3}
$$

如果每个相关历史上都得到非负的精确局部收益，且新策略真的实现合法 affine 目标，就得到 $J(q)\ge J(p)$。例如精确目标式 (J4) 的可行集合包含零时，最优净目标非负，故 $\mathbf w^\top\mathbf B\ge\tau K\ge0$。

### 18.2 为什么这不是当前训练的自动保证

式 (P2) 的历史分布是**新策略** $d_k^q$。旧 rollout 中访问过的前缀收益非负，并不覆盖更新后新访问的所有历史。再加上 $\mathbf B$ 的估计、hindsight gate、对角近似、有限强度与共享参数误差，实际训练不能直接引用式 (P3) 宣称单调改进。

可以把实现误差写得更具体。设最终 Student 为 $\widetilde p$，其在历史 $h$ 的动作分布与目标 $q$ 的 total variation 距离不超过 $\epsilon_k(h)$，则

$$
\begin{aligned}
J(\widetilde p)-J(p)
\ge\sum_k\mathbb E_{h\sim d_k^{\widetilde p}}
\left[\mathbf w_k(h)^\top\mathbf B_k(h)
-\operatorname{span}(Q_k^p(h,\cdot))\epsilon_k(h)\right],
\end{aligned}
\tag{P4}
$$

其中 $\operatorname{span}(Q)=\max Q-\min Q$。证明是先在式 (P2) 中加减目标分布 $q$，再用 $|(\widetilde p-q)^\top Q|\le\operatorname{span}(Q)\operatorname{TV}(\widetilde p,q)$。

若只有估计收益，还可再分离

$$
\mathbf w^\top\mathbf B
=\mathbf w^\top\widehat{\mathbf B}
-\mathbf w^\top(\widehat{\mathbf B}-\mathbf B).
\tag{P5}
$$

式 (P4)–(P5) 清楚指出需要什么证据：选择后的真实方向收益、相关历史上的估计误差、目标策略的实现误差。训练内 KL loss 小或 $\widehat B$ 多为正，都不能单独代替这些量。

## 19. 训练伪代码：原控制器基线与两种扩展

### 19.0 基线：保留原 VERPO 平滑控制器的 Agent 训练循环

在前面已经确认核心控制思想相同后，最直接的训练起点是：**多步 Agent rollout + 原有 GRPO advantage + 原有 Fixed／CTR／FEC 方向 + 原平滑 token 权重 + 独立 reference 与 evidence loss**。这一基线不增加收益预测器或 step 激活费；下文 R0 联合控制、R1 条件收益预测作为独立扩展，不能与基线混称。

下面是设计伪代码，不是当前仓库已完成接入的 Agent trainer。假设使用终局 reward，整条轨迹共享同一个组相对 advantage；一个 rollout group 内的轨迹从同一任务实例独立初始化环境。公式示意使用温度 1、完整词表；生产实现应接现有温度／词表 adapter，而非静默改成 full 模式。

```python
# Student 是可训练模型；Teacher 按配置 frozen / snapshot / EMA 更新。
# 固定权重预算、优化器、rollout 数、reduction 等均从已注册配置读取。

for outer_step in range(cfg.max_outer_steps):
    # 1. 每个任务独立采样 M 条完整交互轨迹。
    # 每一步：Student 生成动作 -> 环境执行 -> observation 加入下一步历史。
    # Student 输入不含训练专用 evidence。
    old_actor = snapshot_for_rollout(student)
    groups = rollout_agent_groups(old_actor, tasks, n=cfg.rollout_n)
    # 保存原始 action / observation / terminal reward / 失败与截断状态，
    # policy mask、step id、token id、采样时 logprob、模型和采样版本。
    persist_raw_trajectories(groups)

    # 2. 保留原序列 VERPO 的终局 GRPO advantage。
    for group in groups:
        R = [trajectory.terminal_reward for trajectory in group]
        A = group_standardize(R, eps=cfg.advantage_epsilon)
        for trajectory, advantage in zip(group, A):
            # 广播到本条轨迹所有 step 的策略 token；observation / padding 为零。
            trajectory.token_advantage = broadcast_to_policy_tokens(
                advantage, trajectory.policy_mask
            )

    # 3. 只为 Teacher 构造证据。正负来源服从配置，负池排除目标自身。
    # 无效负池关闭 evidence mask；reference 和 GRPO 仍保留。
    evidence = prepare_target_excluded_evidence(groups, cfg.evidence_rule)
    batch = pack_interactions_with_explicit_predictor_indices(groups, evidence)

    for minibatch in ppo_minibatches(batch, cfg):
        # 4. 当前 Student 前向。真实历史包含此前工具 observation，
        # 只 gather 每个策略 token 生成之前的预测位置。
        logp = student_policy_token_logprobs(student, minibatch)
        p = logp.exp()                     # 当前 Student；保留梯度
        y = minibatch.sampled_policy_token_ids
        A_token = minibatch.token_advantage.detach()

        # 5. 本次 forward/backward 内 Teacher 快照固定，各分支一致。
        with no_grad():
            q0, q_plus, q_minus, available = teacher_replay_aligned(
                teacher, minibatch, cfg.evidence_rule
            )
            # 多负证据先平均概率；各分支显式对齐同一目标 token。
            # unavailable 行使用 q0 作为中性占位分布，并关闭 evidence mask，
            # 避免先产生非法 log/NaN 再试图乘零。占位不表示证据真实可用。

            direction, loss_payload = build_evidence_direction(
                p.detach(), q0, q_plus, q_minus,
                mode=cfg.fixed_ctr_fec, divergence=cfg.fkl_rkl
            )
            center = (p.detach() * direction).sum(vocab_dim)
            zeta = gather(direction, y) - center
            benefit = A_token * zeta
            fisher = (
                p.detach() * (direction - center[..., None]) ** 2
            ).sum(vocab_dim)

            h = maximum(benefit, 0)
            weight = h / (
                h + cfg.cost_alpha * fisher + cfg.cost_epsilon
            )
            policy_mask = minibatch.policy_mask
            evidence_mask = (
                policy_mask & available
                & configured_group_gate(minibatch, cfg)  # 默认不额外筛组
                & configured_rollout_scope(minibatch, cfg)
            )
            weight = where(evidence_mask, weight, 0)

        # 6. GRPO 的分母始终是 rollout 时的旧概率，不是当前 p。
        ratio = exp(gather(logp, y) - minibatch.old_sampled_logprob)
        rl_token_loss = -minimum(
            ratio * A_token,
            clip(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * A_token
        )

        if cfg.fkl_rkl == "FKL":
            ref_token_loss = (q0 * (log(q0) - logp)).sum(vocab_dim)
            evi_token_loss = -weight * (loss_payload * logp).sum(vocab_dim)
            # loss_payload 是停止梯度的 signed Delta。
        else:
            ref_token_loss = (p * (logp - log(q0))).sum(vocab_dim)
            evi_token_loss = -weight * (p * loss_payload).sum(vocab_dim)
            # loss_payload 是停止梯度的 log-density r。
            # 可保留几何路径配分常数用于报告，但它不改变 Student 梯度。

        # 7. 各通道按既定 mask 和 reduction 聚合，不能相互连带关闭。
        loss = reduce(rl_token_loss, policy_mask, cfg.rl_reduction)
        loss += cfg.lambda_ref * reduce(
            ref_token_loss, policy_mask, cfg.ref_reduction
        )
        loss += cfg.lambda_evi * reduce(
            evi_token_loss, evidence_mask, cfg.evi_reduction
        )
        # reduce 对空 mask 返回有限的零；不能用非零 weight 数作临时分母。

        # 8. 内含梯度累积、clipping、AMP skip 判断与 optimizer step。
        updated = backward_and_maybe_step(student, optimizer, loss, cfg)
        if updated:
            # 只在成功 optimizer update 后执行，按配置处理 frozen/snapshot/EMA。
            update_teacher_after_successful_optimizer_step(teacher, student, cfg)

        log_controller_and_training_metrics(
            benefit, fisher, weight, evidence_mask, loss
        )

    evaluate_and_save_raw_outputs_at_registered_intervals(student, cfg)
```

`build_evidence_direction` 的核心可以写成：

```python
def fisher_apply(p, vector):
    return p * (vector - (p * vector).sum(vocab_dim, keepdim=True))


def build_evidence_direction(p, q0, q_plus, q_minus, mode, divergence):
    # 全部输入／中间量在 controller 中停止梯度；以下 q_plus 在 Fixed 时代表 qe。
    base = q0 if mode == "Fixed" else q_minus
    if divergence == "FKL":
        task = q_plus - base
        if mode == "FEC":
            nuisance = (q_plus + q_minus) / 2 - q0
            alpha = dot(task, fisher_apply(p, nuisance)) / (
                dot(nuisance, fisher_apply(p, nuisance)) + fec_epsilon
            )
            task = task - alpha[..., None] * nuisance
        return task, task                # logit 方向 s；FKL loss 使用的 Delta

    task_r = log(q_plus) - log(base)
    if mode == "FEC":
        nuisance_r = (log(q_plus) + log(q_minus)) / 2 - log(q0)
        task_xi = fisher_apply(p, task_r)
        nuisance_xi = fisher_apply(p, nuisance_r)
        alpha = dot(task_xi, fisher_apply(p, nuisance_xi)) / (
            dot(nuisance_xi, fisher_apply(p, nuisance_xi)) + fec_epsilon
        )
        task_r = task_r - alpha[..., None] * nuisance_r
    return fisher_apply(p, task_r), task_r  # logit 方向 s；RKL loss 使用的 r
```

这里 `sum`、`dot`、`gather` 都按有效策略位置逐行操作；真实长序列应分块回放或重算，伪代码的张量写法不要求长期保存完整 token × vocabulary 缓存。RKL 应使用稳定的 Teacher log 概率，代码中的 `log(q)` 仅为数学简写。

**两个 Student 分布必须分清。**此基线与当前仓库的训练计算点一致：rollout logprob 固定用于 PPO ratio；controller 用更新时当前 Student 的停止梯度分布计算，并随更新重算。因此它是经验训练算法，不能把所有更新中的 benefit 都当作固定行为策略下的无偏条件收益。第 10–18 节的严格方向分析若要求固定旧 Student／Teacher，可另行采用 batch 冻结回放和权重，但那是一项需要标明的训练语义变化。

另外，当前 native veRL 的 VERPO 入口（`verl/verl/trainer/distillation/verpo_zpd.py` 中的 `_first_nested_value_per_row`）会取一条 response 的首个 advantage 再广播。这适合上述终局共享 advantage 基线；将来若采用 step return-to-go、GAE 或前缀条件收益，必须保留真实 token／step advantage，不能继续通过该广播入口把差异压平。本次只核对了入口，没有修改训练代码。

与单段 response 相比，此基线新增的主要是**环境交互、完整历史回放、step/token 索引和 observation mask**。Teacher 纠正仍然只在训练 loss 中使用；Student rollout 执行的是 Student 自己的动作，不应把 Teacher 纠正后的动作偷偷替换进已记录轨迹。

### 19.1 R0 联合控制扩展：事后 replay 加权

R0 的目的，是在保持原 VERPO 架构的情况下验证数据和方向链路。它无需新增 Q 网络，但没有自动消除选择偏差。

```python
# 设计伪代码；当前仓库尚未接入这个 Agent 训练循环。
# 第一步冻结行为 Student、Teacher、采样规则与 evidence 构造规则。
trajectories = rollout_agent(old_student, prompts, n=8)
# 保存全部动作、observation、奖励、失败、EOS、mask 和版本信息。

for group in trajectories.by_prompt():
    advantage = leave_one_out_raw_returns(group)  # 纯终局奖励示意
    for trajectory in group:
        for step in trajectory.steps:
            # 旧 Student 概率必须匹配实际采样分布。
            p, teachers = causal_replay(step, old_student, frozen_teacher)
            direction = build_fixed_ctr_or_fec_direction(p, teachers, kl_mode)
            zeta = sampled(direction) - expectation(p, direction)
            cost = expectation(p, centered(direction, p) ** 2)
            empirical_benefit = advantage[trajectory.id] * zeta
            weights = diagonal_control_then_step_fee(empirical_benefit, cost)
            # 记录：这些是当前样本的经验权重，不是已识别的条件真值。
            save_replay_and_control(step, weights, zeta, cost, empirical_benefit)

loss = outcome_loss + lambda_ref * reference_loss
loss += lambda_evi * signed_evidence_loss(stop_gradient(weights))
update_student(loss)
audit_actual_logit_change_kl_and_independent_returns()
```

FKL／RKL 方向应由第 3–5 节对应的接口生成，不能把已乘 advantage 或已有 ZPD gate 的方向再作为原始 $s$。

### 19.2 R1：前缀条件预测与独立选择

先采集 $X,c$ 与只含生成前信息的特征；按 prompt 做交叉拟合，训练 $b_\phi(f(u))$。在评估 fold 上冻结预测器，以预测收益代替单样本 $X$。

```python
folds = split_by_prompt(all_records)  # sibling 必须同 fold
for heldout in folds:
    model = fit_benefit_predictor(records_outside(heldout))
    for record in heldout:
        predicted_b = model(record.pretoken_features)
        weight = prefix_control(predicted_b, record.fisher_cost)
        store_out_of_fold_weight(record, weight)

# 若要生成前 step 预筛选，另用训练外／预先冻结的 V(h) 预测器。
# 不能用 heldout step 的事后总效用回改早期 token gate 后仍称其可预测。
```

交叉拟合降低直接拟合本条噪声再筛选的偏差，但不保证有限样本符号正确，也不保证训练期 Teacher 特征在部署时可得。是否用 Teacher 在线控制、只在训练时蒸馏，需作为不同部署语义报告。

## 20. 实现核对、实验顺序与当前边界

### 20.1 先保证分析的确是产生轨迹的那个分布

如果实际采样为 $p_b=\operatorname{softmax}(z/T+\mathrm{mask})$，则 score 中的均值与 Fisher 都必须对应 $p_b$。若 $s$ 在原始 logits 坐标定义，温度固定时采样 logits 方向为 $s/T$。不能使用温度 1 的概率解释温度 0.7 的采样统计。

动态 top-p／top-k 采样会改变支持；有限干预需固定支持或重新说明所分析策略。**采样截断**与**VERPO loss 的词表 Top-k 近似**是两件事。当前 `topk_truncated=128` 保留全词表 normalizer，仅在选中坐标做部分计算，不重归一化、不增设 tail bucket；它不自动满足本文完整词表恒等式的精确数值形式。[S3；S6；S7]

Teacher 与 Student evidence 前缀长度不同，必须显式对齐预测位置。对满足本文可预测性假设的因果分支，改写未来 suffix 不应改变较早方向；对于另行设计的 hindsight 分支，未来反馈被前置后影响较早方向是预期行为，不能用前一条检查否定它，但也不能据此沿用可预测方向保证。observation 不进入策略 loss，但继续作为输入上下文。错误工具调用、格式错误、超时和截断须保存并分类，不能删掉失败轨迹来改善统计。

### 20.2 按推导链逐段验证，而不是只看最终 reward

| 环节 | 要验证的量 | 能支持的结论 |
|---|---|---|
| Teacher → 方向 | FKL／RKL loss 的有限差分梯度；FEC 投影残差 | 方向坐标与代码一致 |
| 方向 → score／cost | replay logprob、中心化残差、Fisher chain rule | 行为分布与统计量一致 |
| 样本 → 条件收益 | 同历史样本数、预测校准、独立干预收益 | 收益估计具有多大可靠性 |
| 收益 → 选择 | 独立集上的被选收益、随机位置同预算对照 | 选择是否优于纯稀疏／强度变化 |
| 控制 → 参数更新 | 目标／实际方向余弦、实际 KL、off-route drift | 模型实现了多少理想纠正 |
| 局部 → 任务效果 | 等环境交互、Teacher 前向、总算力下的成功率及原始输出审计 | 是否带来可信的 Agent 增益 |

阶段 A 应冻结模型、回放真实轨迹并做小规模独立干预。阶段 B 再对比无 evidence、统一强度、原平滑 ZPD、标量 step、R0 联合、R1 前缀控制和随机位置匹配移动预算。阶段 C 检查长 horizon、工具扰动、恢复失败与新访问历史。

理论稿不等于训练授权。当前交付是完整说明文档；Agent 环境 adapter、收益预测器和生产训练器仍需独立实现及验证。已有数值内核只检查有限问题，不能替代真实 LLM 训练效果。

## 附录 A. 公式接口总表与旧符号对照

| 位置 | 定义 | 在完整链中的职责 |
|---|---|---|
| Teacher contrast | $\Delta=q_e-q_b$ 或 $r=\log q_e-\log q_b$ | evidence 条件改变 |
| FEC | 从 task 中减去 Fisher 回归的 nuisance 分量 | 调整方向内容 |
| 统一 logit 方向 | $s=\Delta$ 或 $s=F(p)r$ | 后续控制的固定输入 |
| 中心化 score | $\zeta=s(y)-\mathbb E_ps$ | 方向的 log 概率导数 |
| 原版经验 benefit | $\widehat b=\widehat A_i\zeta$ | 当前 GRPO surrogate 对齐 |
| Agent 条件 benefit | $b(u)=\mathbb E[(G-\beta)\zeta\mid u]$ | 当前前缀的后续收益导数 |
| token cost | $c=s^\top F(p)s$ | 局部 KL 曲率 |
| step score／benefit | $S=\sum_t\zeta_t$，$B=\mathbb E[GS\mid h]$ | 完整动作方向与收益 |
| 原平滑控制 | $h/(h+a_cc+\epsilon_c)$ | 已实现的经验接受函数 |
| 联合控制 | $\mathbf w^\top\mathbf B-\tau K(\mathbf w)$ | 在同一目标里选择位置与强度 |
| 对角近似 | 式 (D4)–(D6) | 低成本 token 权重与 step 激活 |
| 训练执行 | 式 (L2)–(L4)，权重停梯度 | outcome、anchor 与 correction 合成 |

原稿用 $u_t$ 表示单位 logit 方向，Agent 稿用 $u$ 表示前缀；本文统一**前缀为 $u$、方向为 $s$**。原稿 $G_t$ 表示 Jacobian，Agent 的 $G_k$ 表示回报；本文 Jacobian 改用 $D_t$，回报仍为 $G_k$。FEC 投影系数 $\alpha_{\mathrm F/R}$、成本尺度 $a_c$ 与前缀控制场 $a(u)$ 分别承担不同职责。

## 附录 B. 原文的两个延伸结论如何放进这条主线

### B.1 全局系数的约束解释

若优化 $J(\pi)$ 并约束 $\mathcal K_{\mathrm{ref}}(\pi)\le\delta$，Lagrangian 可写成 $-J+\lambda_{\mathrm{ref}}(\mathcal K_{\mathrm{ref}}-\delta)$，所以 reference 系数具有 KL 约束乘子的解释。实际固定系数训练不自动执行 dual 更新，也不保证约束恰好满足。

$\lambda_{\mathrm{evi}}$ 乘的是 signed correction，它不一定非负，不能直接解释成同一个 KL 约束乘子。若要控制实际 evidence movement，应单独定义相对 old Student 的移动预算。原版成本尺度、全局 evidence 系数、优化器步长也不能互相替代：改变它们会影响不同的路径和目标。[S1 §21]

### B.2 固定状态的解析最优策略，为什么不是 ZPD 权重

对于固定状态、自由概率变量 $\pi$、固定价值向量 $Q$、$\lambda_r>0$，RKL 目标

$$
\min_{\pi\in\Delta}\left[-\sum_v\pi(v)Q(v)
+\lambda_rD_{\mathrm{KL}}(\pi\Vert q^0)
-\lambda_ew\sum_v\pi(v)r(v)\right]
\tag{B1}
$$

加归一化乘子求导，可得

$$
\pi^*(v)\propto q^0(v)\exp\left(\frac{Q(v)+\lambda_ewr(v)}{\lambda_r}\right).
\tag{B2}
$$

它回答“固定权重和价值后，哪个自由分布最优”；不回答权重应该取多少，也不表示神经网络一步训练达到该分布。

对于 FKL，自由分布目标若是 $-\sum_v\pi(v)Q(v)-\sum_va(v)\log\pi(v)$，且所有 $a(v)>0$，则一阶条件给出

$$
\pi^*(v)=\frac{a(v)}{\mu-Q(v)},\qquad
\sum_v\frac{a(v)}{\mu-Q(v)}=1,\qquad \mu>\max_vQ(v).
\tag{B3}
$$

对于合法 FKL target，$a$ 可为非负加权 Teacher；对于解耦 signed CTR／FEC，$a(v)=\lambda_rq^0(v)+\lambda_ew\Delta(v)$ 可能出现负值。若某坐标为负，令该坐标概率趋零可使目标无下界，不能直接套式 (B3)。这说明 signed correction 应配合 anchor、小更新和审计，而不是伪装成始终合法的 Teacher mixture。[S1 §11.7、§23]

## 附录 C. 来源、冲突处理与文件入口

| 来源 | 本文使用内容 | 读取入口 |
|---|---|---|
| S1 原完整推导 | 第 2–11 节、19–23 节的基础链、参数近似和扩展 | [Math 完整推导](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/Math/VERPO_ZPD_FKL_RKL_Complete_Derivation_zh.md) |
| S2 当前论文 | FKL、RKL 精确梯度，FEC 局部对称假设，平滑控制器的工程定位 | [FKL](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/paper/paper_draft/iclr2026/verpo_zpd_iclr2027/sections/04_forward_kl_verpo.tex)、[RKL](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/paper/paper_draft/iclr2026/verpo_zpd_iclr2027/sections/05_reverse_kl_verpo.tex)、[ZPD](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/paper/paper_draft/iclr2026/verpo_zpd_iclr2027/sections/06_zpd_weight.tex)、[工程近似](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/paper/paper_draft/iclr2026/verpo_zpd_iclr2027/sections/07_engineering_approximation.tex)、[FEC](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/paper/paper_draft/iclr2026/verpo_zpd_iclr2027/sections/08_fec.tex) |
| S3 当前实现 | 方向、FEC、梯度路径、权重有效参数及 detach | [verpo_zpd.py](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/risk_aware_opsd/verpo_zpd.py) |
| S4 原 Agent 推导 | 第 4–9 节的 score、step 几何、标量控制与策略改进 | [derivation.md](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/agent%20opsd/derivation.md) |
| S5 联合控制报告与内核 | 多方向优化、step fee、R0／R1、选择偏差和有限核对 | [报告](Agent_VERPO_%E8%81%94%E5%90%88%E6%8E%A7%E5%88%B6%E7%A0%94%E7%A9%B6%E4%B8%8E%E5%AE%9E%E7%8E%B0%E6%8A%A5%E5%91%8A.md)、[参考内核](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/agent%20opsd/agent_verpo_report/agent_verpo_core.py) |
| S6 当前仓库合同 | 活跃设置、Teacher 更新轴、group gate、RKL-FEC 指纹 | [AGENTS.md](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/AGENTS.md) |
| S7 词表近似说明 | 当前 selected-coordinate、全 normalizer、不重归一化语义 | [verpo_truncated_topk.md](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/docs/verpo_truncated_topk.md) |
| S8 阅读结构说明 | 从读者问题按依赖组织推导的写法 | [理论章节结构分析](https://github.com/hamsterjiang23/PGR-Probe/blob/9c01f2bcd05aea1b3741ba60ede9aad201e0875c/docs/verpo_theoretical_section_rhetorical_analysis.tex) |

本次整理发现两处需显式处理的历史差异：

1. 原稿某些段落把 reference 描述为全程固定初始模型；当前合同允许 snapshot／EMA Teacher 在成功更新后刷新，并可供各 Teacher 分支共用。本文使用“本次更新固定、跨更新由协议决定”，不把某个旧实验的冻结设定写成普遍定理。
2. 当前论文 `07_engineering_approximation.tex` 的 Top-k 段落仍描述 tail bucket 与重归一化；当前专门文档和实现采用保留全词表 normalizer 的 selected-coordinate 近似。本文实现描述按 S3／S6／S7，完整数学推导仍按完整支持；本次未修改旧论文或训练代码。

复核入口：`../verify_derivation.py` 检查原 Agent 有限推导，`verify_agent_verpo.py` 检查联合内核。它们支持有限例子里的代数与数值身份；不验证真实任务收益、收益预测可靠性或目标硬件上的训练行为。

本次检查：全文 96 个编号公式、474 个行内公式通过 Markdown／KaTeX 解析；13 张表格和 16 个文件链接通过结构检查。基础桥接部分 17 项有限 CPU 代数／梯度检查、原 Agent 11 项有限枚举检查、联合内核 17 项数值检查全部通过。HTML 中的代表性推导与控制器对照已目视检查；这些检查不构成完整形式化证明或真实训练性能验证。
