# 理论附注：椭圆分布下 ES 比值的结构性质

本附注把 [README](README.md) 中「一个可以先写下来的事实」整理成带假设、证明与适用边界的命题。
命题 1 与推论 2 是已知结果（见文末文献），这里按本篇的记号重新陈述并给出完整证明；
命题 2 刻画性质在哪些情形下失效，其中均值项的精确条件（(i)）与本篇的实验设定直接相关。

## 0 · 记号

- $X\in\mathbb R^d$：资产收益向量（本篇 $d=6$）；组合 $w\in\mathbb R^d$ 的**损失** $L_w=-w^\top X$。
- 置信水平 $q\in(0,1)$（本篇 $q=0.95$）。对 $\mathbb E|L|<\infty$ 的损失，
  $$\mathrm{VaR}_u(L)=F_L^{-1}(u)=\inf\{x:F_L(x)\ge u\},\qquad
    \mathrm{ES}_q(L)=\frac1{1-q}\int_q^1\mathrm{VaR}_u(L)\,du .$$
  `es_sample` 是它的经验版本（最差 $\lceil(1-q)n\rceil$ 个观测的平均损失）。
- $\sigma_w=\sqrt{w^\top\Sigma w}$。

**定义（椭圆分布）。** 称 $X\sim E_d(\mu,\Sigma,\psi)$，若存在 $\mu\in\mathbb R^d$、半正定 $\Sigma\neq0$ 与函数 $\psi:[0,\infty)\to\mathbb R$，使
$$\varphi_{X-\mu}(t)=\mathbb E\,e^{\,i t^\top (X-\mu)}=\psi\!\left(t^\top\Sigma t\right),\qquad t\in\mathbb R^d .$$
$\Sigma$ 称为离散矩阵，$\psi$ 称为特征生成元。$(\Sigma,\psi)$ 只在 $(c\Sigma,\ \psi(\cdot/c))$ 的意义下可识别。
若二阶矩存在，则 $\mathrm{Cov}(X)=-2\psi'(0)\,\Sigma$，即协方差与离散矩阵只差一个正常数。
高斯：$\psi(s)=e^{-s/2}$；多元 $t_\nu$：正态方差混合 $X=\mu+\sqrt{W}\,Z$，$Z\sim N(0,\Sigma)$，$W\sim \nu/\chi^2_\nu$（本篇的 `draw_innov` 再乘 $\sqrt{(\nu-2)/\nu}$ 把协方差归一）。

**假设。**
- **(A1)** $X\sim E_d(\mu,\Sigma,\psi)$，且 $X$ 非退化（$\Sigma\ne0$）。
- **(A2)** $\mathbb E\|X\|<\infty$（ES 有定义；对 $t_\nu$ 即 $\nu>1$。本篇用单位方差的 $t(4)$，需要 $\nu>2$）。
- **(A3)** 损失是组合收益的线性函数：$L_w=-w^\top X$。

## 1 · 两个引理

**引理 1（线性组合的分布）。** 在 (A1) 下，存在一个不依赖 $w$ 的对称随机变量 $Z$，其特征函数为 $\psi(s^2)$，使得对任意 $w$，
$$L_w\ \overset{d}{=}\ -w^\top\mu+\sigma_w Z .$$

*证明。* 对 $s\in\mathbb R$，
$\varphi_{w^\top(X-\mu)}(s)=\varphi_{X-\mu}(sw)=\psi\!\left(s^2\,w^\top\Sigma w\right)=\psi(s^2\sigma_w^2)$。
因为 $\Sigma\ne0$，可取 $w_0$ 使 $\sigma_{w_0}=1$，令 $Z:=w_0^\top(X-\mu)$，则 $\varphi_Z(s)=\psi(s^2)$。
于是当 $\sigma_w>0$ 时，$\varphi_{\sigma_w Z}(s)=\psi(s^2\sigma_w^2)=\varphi_{w^\top(X-\mu)}(s)$，由特征函数唯一性得
$w^\top(X-\mu)\overset d=\sigma_w Z$；当 $\sigma_w=0$ 时两边都是 $0$（特征函数恒为 $\psi(0)=1$）。
$\varphi_Z$ 是实的偶函数，所以 $Z\overset d=-Z$，从而 $-w^\top(X-\mu)\overset d=\sigma_w Z$。$\square$

**引理 2（ES 的性质）。** 设 $\mathbb E|Y|<\infty$。
1. （法则不变）若 $Y\overset d=Y'$，则 $\mathrm{ES}_q(Y)=\mathrm{ES}_q(Y')$。
2. （平移与正齐次）对 $a\in\mathbb R,\ b\ge0$，$\mathrm{ES}_q(a+bY)=a+b\,\mathrm{ES}_q(Y)$。
3. （严格正）若 $Y$ 非退化且 $\mathbb E Y=0$，则 $\mathrm{ES}_q(Y)>0$。

*证明。* (1) ES 只依赖分位数函数，而分位数函数只依赖分布。
(2) $b>0$ 时 $x\mapsto a+bx$ 严格增，故 $\mathrm{VaR}_u(a+bY)=a+b\,\mathrm{VaR}_u(Y)$，积分即得；$b=0$ 时 $a+bY\equiv a$，$\mathrm{ES}_q(a)=a$。
(3) 记 $f(u)=\mathrm{VaR}_u(Y)$，它单调不减，且 $\int_0^1 f=\mathbb EY=0$。
设 $\bar f_-=\tfrac1q\int_0^q f$，$\bar f_+=\tfrac1{1-q}\int_q^1 f=\mathrm{ES}_q(Y)$。由单调性 $\bar f_-\le f(q^-)\le f(q^+)\le\bar f_+$，
又 $q\bar f_-+(1-q)\bar f_+=0$，故 $\bar f_+\ge0$。若 $\bar f_+=0$，则 $\bar f_-=0$，上面的不等式全部取等号，
$f$ 在 $(0,1)$ 上几乎处处为常数 $0$，即 $Y\equiv0$，与非退化矛盾。$\square$

## 2 · 主命题

**命题 1（椭圆分布下 ES 的线性表示与比值）。** 在 (A1)–(A3) 下，令 $k_q:=\mathrm{ES}_q(Z)$，其中 $Z$ 为引理 1 中的变量。则

**(a)** $k_q>0$，且只依赖 $(q,\psi)$，不依赖 $w$、$\mu$；对任意 $w$，
$$\mathrm{ES}_q(L_w)=-w^\top\mu+k_q\,\sigma_w .$$

**(b)** 若 $\mu=0$，则对任意满足 $\sigma_{w_2}>0$ 的 $w_1,w_2$，
$$\theta:=\frac{\mathrm{ES}_q(L_{w_1})}{\mathrm{ES}_q(L_{w_2})}=\sqrt{\frac{w_1^\top\Sigma w_1}{w_2^\top\Sigma w_2}} .$$

**(c)** （比值的不变性）在 (b) 中，$\theta$ 不依赖生成元 $\psi$，也不依赖置信水平 $q$；把 $\Sigma$ 换成任意 $c\Sigma$（$c>0$）——特别是换成 $\mathrm{Cov}(X)$——时 $\theta$ 不变。
因此，$\theta$ 是离散矩阵**形状** $\Sigma/\mathrm{tr}\,\Sigma$ 的泛函。

*证明。* (a) 由引理 1 与引理 2(1)(2)：$\mathrm{ES}_q(L_w)=\mathrm{ES}_q(-w^\top\mu+\sigma_wZ)=-w^\top\mu+\sigma_w\,\mathrm{ES}_q(Z)$。
$Z$ 对称且可积，所以 $\mathbb EZ=0$；由 (A1)，$Z$ 非退化，引理 2(3) 给出 $k_q>0$。$Z$ 的分布只由 $\psi$ 决定，故 $k_q$ 只依赖 $(q,\psi)$。

(b) 由 (a)，$\mathrm{ES}_q(L_{w_2})=k_q\sigma_{w_2}>0$，分母非零；$\theta=k_q\sigma_{w_1}/(k_q\sigma_{w_2})=\sigma_{w_1}/\sigma_{w_2}$。

(c) 右端不含 $\psi$ 与 $q$；把 $\Sigma$ 换成 $c\Sigma$，分子分母同乘 $\sqrt c$。$\square$

**推论 1（插件估计量对分布假设不敏感）。** 设 $\hat\Sigma$ 是任意离散矩阵的估计（例如样本协方差），$\hat k$ 是任意正常数。插件估计
$\widehat{\mathrm{ES}}(w)=\hat k\sqrt{w^\top\hat\Sigma w}$ 给出的比值
$$\hat\theta=\sqrt{\frac{w_1^\top\hat\Sigma w_1}{w_2^\top\hat\Sigma w_2}}$$
与 $\hat k$ 无关：用高斯常数 $k^{N}_q$ 还是 $t$ 常数 $k^{t_\nu}_q$，得到的比值完全相同；而且 $\hat\Sigma$ 只需要估计对形状。
相反，**水平**的总体偏差是常数 $k^{N}_q/k^{t_\nu}_q-1$。本篇 $q=0.95,\ \nu=4$ 时
$k^N=2.0627,\ k^{t_4}=2.2648$，偏差为 $-8.92\%$——即 README 中「高斯插件对 ES 水平稳定低估约 9%」的总体值。

**推论 2（决策等价）。** 设 $\mu=0$，可行集 $\mathcal W\subseteq\{w:\mathbf 1^\top w=1\}$ 任意（例如本篇的单资产上限 40%）。
1. $\arg\min_{w\in\mathcal W}\mathrm{ES}_q(L_w)=\arg\min_{w\in\mathcal W}w^\top\Sigma w$。
2. 在 $\sigma_w>0$ 处，ES 的 Euler 风险贡献 $RC_i(w)=w_i\,\partial\mathrm{ES}_q(L_w)/\partial w_i$ 满足
   $$\frac{RC_i(w)}{\mathrm{ES}_q(L_w)}=\frac{w_i(\Sigma w)_i}{w^\top\Sigma w},$$
   即 ES 贡献份额等于方差贡献份额。所以 ES 平价 $\Leftrightarrow$ 波动率平价（ERC）。

*证明。* (1) 由命题 1(a)，$\mathrm{ES}_q(L_w)=k_q\sqrt{w^\top\Sigma w}$，$k_q>0$，$x\mapsto k_q\sqrt x$ 严格增。
(2) $\partial\sigma_w/\partial w_i=(\Sigma w)_i/\sigma_w$，所以 $RC_i=k_q\,w_i(\Sigma w)_i/\sigma_w$，除以 $\mathrm{ES}_q(L_w)=k_q\sigma_w$ 即得。
（$\sum_iRC_i=\mathrm{ES}_q(L_w)$，这是一阶正齐次函数的 Euler 定理。）$\square$

推论 2 说明：在椭圆族中，ES 水平误差（$\hat k\ne k_q$）对两类决策都**没有影响**，影响决策的只有 $\hat\Sigma$ 的形状误差。
这就是本篇把估计误差拆成「水平」与「形状」的理论依据。

## 3 · 适用边界

**命题 2（性质失效的情形）。**

**(i) 非零均值。** 在 (A1)–(A3) 下，记 $m_j=w_j^\top\mu$，$SR_j=m_j/\sigma_{w_j}$，并设 $k_q\sigma_{w_2}>m_2$。则
$$\theta_\mu=\frac{k_q\sigma_{w_1}-m_1}{k_q\sigma_{w_2}-m_2}
=\frac{\sigma_{w_1}}{\sigma_{w_2}}\cdot\frac{1-SR_1/k_q}{1-SR_2/k_q},$$
并且 $\theta_\mu=\sigma_{w_1}/\sigma_{w_2}$ **当且仅当** $SR_1=SR_2$。否则 $\theta_\mu$ 依赖 $k_q$，从而依赖 $\psi$：分布误设重新进入比值。
相对偏差为
$$\frac{\theta_\mu}{\sigma_{w_1}/\sigma_{w_2}}-1=\frac{SR_2-SR_1}{k_q-SR_2}\approx\frac{SR_2-SR_1}{k_q}.$$

*证明。* 第一个等式由命题 1(a) 直接得到；提出 $k_q\sigma_{w_j}$ 得第二个等式。
$\theta_\mu=\sigma_{w_1}/\sigma_{w_2}\iff(1-SR_1/k_q)=(1-SR_2/k_q)\iff SR_1=SR_2$。偏差式由第二个等式移项得到。$\square$

*与本篇的联系。* 实验 A 均值为 0，比值精确成立。实验 B 设 $\mu_i=0.3\,\sigma_i^{\text{年}}/244$：每个**资产**的夏普相同，
但组合的夏普 $w^\top\mu/\sigma_w$ 一般不同，所以比值不精确。日夏普在 $0.3/\sqrt{244}\approx0.02$ 量级，$k_q\approx2.26$，
偏差 $\approx(SR_2-SR_1)/k_q$ 只有千分之几，在实验 B 的精度下可以忽略。

**(ii) 非椭圆分布：即使均值为零、协方差正确，也不保证成立。** 反例：$X=(X_1,X_2)$ 相互独立，$X_1\sim N(0,1)$，
$X_2$ 以各 1/2 的概率取 $\pm1$。则 $\mathbb EX=0$，$\mathrm{Cov}(X)=I$，$\sigma_{e_1}=\sigma_{e_2}=1$，命题 1 的公式给出 $\theta=1$；但
$\mathrm{ES}_{0.95}(-X_1)=k^N_{0.95}=2.063$，$\mathrm{ES}_{0.95}(-X_2)=1$，真实比值为 $2.063$。
本篇的崩盘混合是同一现象在 6 维的例子：协方差严格等于高斯与 $t(4)$ 的协方差，但高斯插件有不随样本消失的 $+13\%$ 形状偏差。

一般地，对给定的 $X$（$\mathbb EX=0$，协方差 $\Sigma$）与给定的 $q$，「比值等于 $\sigma_{w_1}/\sigma_{w_2}$ 对所有组合对成立」
等价于 $w\mapsto\mathrm{ES}_q(L_w)/\sigma_w$ 在 $\{\sigma_w>0\}$ 上为常数。这是一个只涉及单个 $q$ 的条件，比椭圆性弱；
若对所有 $q$ 都成立，则所有 $w^\top X$ 的分位数函数两两成比例。这一条件与椭圆性的精确关系，留作后续核对文献。

**(iii) 非线性损失。** 若 $L=h(X)$ 不是 $X$ 的线性函数（例如含期权的组合），引理 1 不再适用，ES 一般没有 $k\,\sigma$ 形式，比值也没有。

**(iv) 退化的分母。** 若 $\sigma_{w_2}=0$（例如组合恰好对冲掉全部风险），则 $\mathrm{ES}_q(L_{w_2})=-w_2^\top\mu$，$\mu=0$ 时为 0，比值无定义。

| 条件 | 比值 $=\sigma_{w_1}/\sigma_{w_2}$ | 依据 |
|---|---|---|
| 零均值椭圆 | 成立 | 命题 1(b) |
| 高斯 ↔ $t_\nu$ 之间误设，$\Sigma$ 形状正确 | 成立 | 命题 1(c)、推论 1 |
| $\Sigma$ 整体差一个正常数 | 成立 | 命题 1(c) |
| 非零均值 | 当且仅当两组合夏普相同 | 命题 2(i) |
| 非椭圆（零均值、协方差正确） | 不保证 | 命题 2(ii) 反例、崩盘混合 |
| 非线性损失 | 不保证 | 命题 2(iii) |

## 4 · 数值核对

用本篇的资产宇宙（`SIGMA`、`Q = 0.95`、单位方差 $t(4)$），400 万天模拟，$w_1=(0.05,0,0.15,0.40,0.25,0.15)$，$w_2$ 为等权：

| 核对项 | 模拟 | 理论 |
|---|---|---|
| $t(4)$ 下 ES 比值 | 0.64478 | $\sigma_{w_1}/\sigma_{w_2}=0.64472$ |
| 非零均值（均值放大 20 倍以便看清） | 0.63217 | 命题 2(i)：0.63212（$\sqrt{\cdot}$ 公式给出 0.64472） |
| $w_1$ 的 ES 贡献份额 vs 方差贡献份额 | 0.054 / 0 / 0.174 / 0.011 / 0.292 / 0.470 | 0.054 / 0 / 0.174 / 0.011 / 0.292 / 0.469 |
| 高斯插件的总体水平偏差 | — | $k^N/k^{t_4}-1=-8.92\%$ |

差异都在 Monte Carlo 误差内。

## 5 · 下一步

命题 1 把估计目标从「ES」换成了「$\Sigma$ 的形状泛函」$g(\Sigma)=\sqrt{w_1^\top\Sigma w_1/w_2^\top\Sigma w_2}$。
由此自然引出：(1) 用 Delta 方法求 $g(\hat\Sigma)$ 的渐近方差（$g$ 对尺度不变，$\nabla g$ 与 $\mathrm{vec}\,\Sigma$ 正交，峰度的尺度项会消去）；
(2) 只估计形状、不需要矩条件的 Tyler M 估计量。

## 文献

- Fang, K.-T., Kotz, S. & Ng, K. W. (1990). *Symmetric Multivariate and Related Distributions*. Chapman & Hall.（椭圆分布的定义与线性组合的性质）
- McNeil, A. J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management*, 2nd ed. Princeton University Press.（椭圆分布下的 VaR/ES 与 Euler 分配）
- Acerbi, C. & Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance* 26(7).（ES 的定义与性质）
- Tasche, D. (1999). Risk contributions and performance measurement. Working paper, TU München.（Euler 风险贡献）
