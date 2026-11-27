# 理论附注（二）：ES 比值估计量的一致性与渐近分布

接 [theory_es_ratio.md](theory_es_ratio.md)（命题 1–2）。命题 1 把估计目标化为离散矩阵形状的泛函
$g(\Sigma)=\sqrt{w_1^\top\Sigma w_1/w_2^\top\Sigma w_2}$。本附注研究三种估计量——样本协方差插件、Tyler 插件、历史法（HS）——的一致性与渐近分布，
并给出样本协方差插件在四阶矩不存在时的收敛速度与极限分布。

各结论的来源：
- 命题 3(a)、命题 4、推论 4.1、推论 5.1、命题 7 是这里完整证明的，只引用标准工具（强大数定律、多元 CLT、Delta 方法、Breiman 引理、广义中心极限定理）；
- Tyler 估计量的一致性与渐近协方差（命题 3(b)、5(a)），以及 HS 的渐近正态性（命题 6），引用文献中的定理；
- 命题 5(b)（样本均值中心化不影响 Tyler 的渐近分布）只给出证明要点，未处理一致性条件，由模拟支持。

数值核对见 [asymptotics_check.ipynb](asymptotics_check.ipynb) 与 [bias_variance.ipynb](bias_variance.ipynb)。

## 0 · 设定与记号

- $X_1,\dots,X_T$ 独立同分布，取值于 $\mathbb R^d$（$d\ge2$），$\mu=\mathbb EX$，$Y=X-\mu$，$\Sigma=\mathrm{Cov}(X)$。
- 两个组合 $w_1,w_2$，$\sigma_j^2=w_j^\top\Sigma w_j>0$，$L_j=w_j^\top Y$，$U_j=L_j/\sigma_j$，$\rho=\mathrm{Corr}(L_1,L_2)$，$|\rho|<1$。
- 样本协方差 $S_T=\frac1{T-1}\sum_t(X_t-\bar X)(X_t-\bar X)^\top$。
- $g(M)=\sqrt{w_1^\top Mw_1/w_2^\top Mw_2}$，定义在 $\{M\text{ 对称}:w_1^\top Mw_1>0,\ w_2^\top Mw_2>0\}$ 上。记 $\theta_G=g(\Sigma)=\sigma_1/\sigma_2$。
  由命题 1，零均值椭圆分布下 $\theta_G=\theta$（真实 ES 比值）；一般情形下 $\theta_G$ 是样本协方差插件的**伪真值**，$\theta_G/\theta-1$ 即 bias_variance.ipynb 中的渐近偏差 β。
- 三种估计量：$\hat\theta_G=g(S_T)$；$\hat\theta_{\mathrm{Ty}}=g(\hat V_T)$，$\hat V_T$ 为 Tyler 形状估计量；$\hat\theta_{HS}=\widehat{\mathrm{ES}}_1/\widehat{\mathrm{ES}}_2$，$\widehat{\mathrm{ES}}_j$ 为最差 $\lceil(1-q)T\rceil$ 个损失的平均。

**引理 A（$g$ 的性质）。** (i) $g(cM)=g(M)$，$c>0$。(ii) $g$ 在定义域内光滑，对对称矩阵 $H$，
$$Dg(M)[H]=\operatorname{tr}(G_MH),\qquad G_M=\frac{g(M)}2\left(\frac{w_1w_1^\top}{w_1^\top Mw_1}-\frac{w_2w_2^\top}{w_2^\top Mw_2}\right).$$
(iii) $\operatorname{tr}(G_MM)=0$。(iv) $\operatorname{tr}(G_\Sigma\Sigma G_\Sigma\Sigma)=\tfrac12\theta_G^2(1-\rho^2)$。

*证明。* (i) 分子分母同乘 $c$。(ii) $\log g(M)=\tfrac12\log(w_1^\top Mw_1)-\tfrac12\log(w_2^\top Mw_2)$，对 $M$ 沿 $H$ 求导得
$\tfrac12(w_1^\top Hw_1/w_1^\top Mw_1-w_2^\top Hw_2/w_2^\top Mw_2)$，乘以 $g(M)$，并用 $w^\top Hw=\operatorname{tr}(ww^\top H)$。
(iii) 由 (ii) 直接计算，$\tfrac{g}{2}(1-1)=0$；也可由 (i) 对 $c$ 求导得到。
(iv) 记 $A=w_1w_1^\top$，$B=w_2w_2^\top$。$\operatorname{tr}(A\Sigma A\Sigma)=\sigma_1^4$，$\operatorname{tr}(B\Sigma B\Sigma)=\sigma_2^4$，
$\operatorname{tr}(A\Sigma B\Sigma)=(w_1^\top\Sigma w_2)^2=\rho^2\sigma_1^2\sigma_2^2$，所以
$\operatorname{tr}(G\Sigma G\Sigma)=\frac{\theta_G^2}4(1+1-2\rho^2)$。$\square$

## 1 · 一致性

**命题 3。**

**(a)** 若 $\mathbb E\|X\|^2<\infty$，则 $\hat\theta_G\to\theta_G$ a.s.。不需要椭圆假设；在零均值椭圆分布下 $\theta_G=\theta$。

**(b)** 若 $X\sim E_d(\mu,\Sigma,\psi)$ 绝对连续、$\Sigma$ 正定，中心 $\mu$ 已知且 $T>d$，则 Tyler 不动点方程 a.s. 存在在尺度意义下唯一的解，且 $\hat V_T\to c\Sigma$ a.s.（$c>0$ 由归一化决定）
（Tyler 1987，其中也证明了不动点迭代收敛）。因此 $\hat\theta_{\mathrm{Ty}}\to\theta_G$ a.s.，**不需要任何矩条件**。

**(c)** 若 $\mathbb E|L_j|<\infty$，则 $\widehat{\mathrm{ES}}_j\to\mathrm{ES}_q(L_j)$ a.s.（Acerbi & Tasche 2002），从而 $\hat\theta_{HS}\to\theta$ a.s.，对任意分布成立。

*证明。* (a) 由强大数定律，$\bar X\to\mu$，$\frac1T\sum_tX_tX_t^\top\to\mathbb EXX^\top$ a.s.，故
$S_T=\frac T{T-1}\big(\frac1T\sum_tX_tX_t^\top-\bar X\bar X^\top\big)\to\Sigma$ a.s.。$g$ 在 $\Sigma$ 处连续（$\sigma_j>0$），由连续映射定理得结论。
(b) 引用的定理给出 $\hat V_T\to c\Sigma$；由引理 A(i)，$g(\hat V_T)\to g(c\Sigma)=g(\Sigma)$。
(c) 两个 ES 估计量各自 a.s. 收敛，分母极限 $\mathrm{ES}_q(L_2)>0$（引理 2(3)，theory_es_ratio.md），由连续映射定理得结论。$\square$

注意 (a) 与 (b) 的极限都是 $g(\Sigma)$，与分布无关；只有在椭圆族中它才等于真实 ES 比值。非椭圆时，Tyler 的极限一般也不是 $c\Sigma$，
崩盘混合下它的极限比值偏离真值 +27%（q = 0.95），比样本协方差插件的 +9% 更远（bias_variance.ipynb 第 5.4 节）。

## 2 · 渐近正态性

**命题 4（样本协方差插件）。** 若 $\mathbb E\|X\|^4<\infty$，则
$$\sqrt T\left(\frac{\hat\theta_G}{\theta_G}-1\right)\xrightarrow d N(0,V_G),\qquad V_G=\tfrac14\operatorname{Var}\!\left(U_1^2-U_2^2\right).$$
不需要椭圆假设。

*证明。* **第 1 步（协方差的 CLT）。** $S_T=\frac T{T-1}\big(\frac1T\sum_tY_tY_t^\top-\bar Y\bar Y^\top\big)$，所以
$$\sqrt T(S_T-\Sigma)=\frac1{\sqrt T}\sum_t(Y_tY_t^\top-\Sigma)+R_T,\qquad
R_T=-\tfrac{T}{T-1}\sqrt T\,\bar Y\bar Y^\top+\tfrac{\sqrt T}{T-1}\Big(\tfrac1T\textstyle\sum_tY_tY_t^\top-\bar Y\bar Y^\top\Big).$$
$\sqrt T\bar Y=O_p(1)$，$\bar Y=o_p(1)$，所以第一项 $o_p(1)$；第二项 $O_p(T^{-1/2})$。$\mathrm{vec}(YY^\top)$ 有有限二阶矩（因 $\mathbb E\|Y\|^4<\infty$），
由多元 CLT，$\sqrt T\,\mathrm{vec}(S_T-\Sigma)\xrightarrow dN(0,\Gamma)$，$\Gamma=\mathrm{Cov}(\mathrm{vec}\,YY^\top)$。

**第 2 步（Delta 方法）。** 由引理 A(ii)，$\sqrt T(\hat\theta_G-\theta_G)\xrightarrow dN(0,\ \mathrm{vec}(G)^\top\Gamma\,\mathrm{vec}(G))$，$G=G_\Sigma$。而
$$\mathrm{vec}(G)^\top\Gamma\,\mathrm{vec}(G)=\operatorname{Var}\big(\operatorname{tr}(GYY^\top)\big)
=\operatorname{Var}\Big(\frac{\theta_G}2\big(U_1^2-U_2^2\big)\Big),$$
因为 $\operatorname{tr}(w_jw_j^\top YY^\top)/\sigma_j^2=U_j^2$。两边除以 $\theta_G^2$ 得结论。$\square$

**推论 4.1（椭圆分布）。** 若 $X$ 为椭圆分布且 $\mathbb E\|X\|^4<\infty$，记峰度参数 $\kappa=\mathbb EU^4/3-1$（对所有方向相同），则
$$V_G=(1+\kappa)(1-\rho^2).$$
单位方差 $t_\nu$（$\nu>4$）：$1+\kappa=(\nu-2)/(\nu-4)$；高斯：$V_G=1-\rho^2$。

*证明。* 椭圆分布的四阶矩满足 $\mathbb E[Y_iY_jY_kY_l]=(1+\kappa)(\sigma_{ij}\sigma_{kl}+\sigma_{ik}\sigma_{jl}+\sigma_{il}\sigma_{jk})$（Muirhead 1982，第 1 章关于椭圆分布的部分）。
应用于 $(U_1,U_2)$（单位方差、相关系数 ρ）：$\mathbb EU_j^4=3(1+\kappa)$，$\mathbb EU_1^2U_2^2=(1+\kappa)(1+2\rho^2)$。于是
$$\operatorname{Var}(U_1^2-U_2^2)=6(1+\kappa)-2(1+\kappa)(1+2\rho^2)-(\mathbb EU_1^2-\mathbb EU_2^2)^2=4(1+\kappa)(1-\rho^2).$$
对 $t_\nu$，$\mathbb EU^4=3(\nu-2)/(\nu-4)$。$\square$

用 $\Gamma$ 的显式形式 $\Gamma=(1+\kappa)(I+K)(\Sigma\otimes\Sigma)+\kappa\,\mathrm{vec}\Sigma\,\mathrm{vec}\Sigma^\top$（$K$ 为交换矩阵）同样可以得到这个结果：
第二项因引理 A(iii) 消失——比值对尺度不变，所以峰度中「整体尺度」的那部分不进入渐近方差；第一项由引理 A(iv) 给出 $(1+\kappa)\theta_G^2(1-\rho^2)$。

**命题 5（Tyler 插件）。** 设 $X\sim E_d(\mu,\Sigma,\psi)$ 绝对连续、$\Sigma$ 正定。

**(a)**（中心已知）
$$\sqrt T\left(\frac{\hat\theta_{\mathrm{Ty}}}{\theta}-1\right)\xrightarrow d N(0,V_{\mathrm{Ty}}),\qquad V_{\mathrm{Ty}}=\frac{d+2}d\,(1-\rho^2),$$
对所有椭圆分布相同，不需要矩条件。

*证明。* Tyler（1987）证明了：存在 $c>0$ 与常数 $\sigma_2$，使
$\sqrt T\,\mathrm{vec}(\hat V_T-c\Sigma)\xrightarrow dN\big(0,\ \tfrac{d+2}d(I+K)(c\Sigma\otimes c\Sigma)+\sigma_2\,\mathrm{vec}\Sigma\,\mathrm{vec}\Sigma^\top\big)$，
其中第一项的系数 $(d+2)/d$ 与径向分布无关，$\sigma_2$ 只依赖归一化方式。由 Delta 方法，渐近方差为
$\mathrm{vec}(G_{c\Sigma})^\top[\cdots]\mathrm{vec}(G_{c\Sigma})$。$G_{c\Sigma}=G_\Sigma/c$，第二项因引理 A(iii) 为 0；
第一项为 $\tfrac{d+2}d\cdot2\operatorname{tr}(G_\Sigma\Sigma G_\Sigma\Sigma)=\tfrac{d+2}d\theta^2(1-\rho^2)$（引理 A(iv)，并用 $(I+K)$ 作用于对称矩阵的 vec 等于乘 2）。$\square$

**(b)**（样本均值中心化，证明要点）若另有 $\mathbb E\|X\|^2<\infty$，用 $\bar X$ 代替 $\mu$ 后 (a) 的结论不变。

*要点。* $\hat V_T$ 是估计方程 $\frac1T\sum_t\psi(X_t-m,V)=0$ 的解，$\psi(y,V)=d\,yy^\top/(y^\top V^{-1}y)-V$。
$\psi$ 中依赖 $y$ 的部分是 $y$ 的零次齐次偶函数，它对 $y$ 的导数是 $-1$ 次齐次的**奇**函数；椭圆分布关于 $\mu$ 对称，且 $d\ge2$ 时 $\mathbb E\|Y\|^{-1}<\infty$，
所以 $\mathbb E[\partial_m\psi(Y,V)]=0$。于是 $m$ 的 $\sqrt T$ 阶估计误差（$\sqrt T(\bar X-\mu)=O_p(1)$）对 $\hat V_T$ 没有一阶影响。
严格证明还需要对 $m$ 在 $\mu$ 附近一致的随机等度连续性，这里没有处理。
bias_variance.ipynb 第 5.2 节中，用样本均值中心化的 Tyler 比值 T·Var 在所有椭圆 DGP 下为 0.57–0.64，与 $V_{\mathrm{Ty}}=0.60$ 一致。

**推论 5.1（Tyler 与样本协方差的渐近效率）。** 对单位方差 $t_\nu$，$\nu>4$：
$$\frac{V_{\mathrm{Ty}}}{V_G}=\frac{(d+2)(\nu-4)}{d(\nu-2)},\qquad V_{\mathrm{Ty}}<V_G\iff\nu<d+4 .$$
高斯（$\nu\to\infty$）时比值为 $(d+2)/d$，即 Tyler 的效率为 $d/(d+2)$。$2<\nu\le4$ 时 Tyler 仍是 $\sqrt T$ 收敛，样本协方差插件更慢（命题 7）。

*证明。* 由推论 4.1 与命题 5(a)，$1-\rho^2$ 约去。$\nu>4$ 时两边乘以正数 $d(\nu-2)$：
$(d+2)(\nu-4)<d(\nu-2)\iff2\nu<2d+8\iff\nu<d+4$。$\square$

本篇 $d=6$，交叉点 $\nu=10$。bias_variance.ipynb 第 5.3 节：t(3)–t(8) 下 Tyler 在所有 T 上显著更好，t(15) 与高斯下样本协方差显著更好。

**命题 6（HS，引用）。** 若 $\mathbb EL_j^2<\infty$，$F_{L_j}$ 在 $v_j=\mathrm{VaR}_q(L_j)$ 处可导且密度为正，则（Chen 2008）
$\widehat{\mathrm{ES}}_j-\mathrm{ES}_j=\frac1T\sum_t\mathrm{IF}_j(X_t)+o_p(T^{-1/2})$，
$\mathrm{IF}_j=v_j+(L_j-v_j)_+/(1-q)-\mathrm{ES}_j$。对两个组合联合应用并用 Delta 方法，
$$\sqrt T\left(\frac{\hat\theta_{HS}}\theta-1\right)\xrightarrow dN(0,V_{HS}),\qquad V_{HS}=\operatorname{Var}\!\left(\frac{\mathrm{IF}_1}{\mathrm{ES}_1}-\frac{\mathrm{IF}_2}{\mathrm{ES}_2}\right).$$
对 $t_\nu$ 只需 $\nu>2$。

## 3 · 四阶矩不存在时：收敛速度与稳定极限

**命题 7。** 设 $X_t=\mu+\sqrt{W_t}\,Z_t$，$Z_t\sim N(0,\Sigma_0)$，$W_t>0$ 与 $Z_t$ 独立，且 $P(W>x)\sim c\,x^{-\alpha}$（$x\to\infty$）。
单位方差的 $t_\nu$ 属于这一类：$W=(\nu-2)/\chi^2_\nu$，$\alpha=\nu/2$（因为 $P(\chi^2_\nu<y)\asymp y^{\nu/2}$，$y\to0$）。$\Sigma=\mathbb E[W]\Sigma_0$。

**(a)** 若 $1<\alpha<2$（即 $2<\nu<4$），则存在 $a_T=c'T^{1/\alpha}$，使
$$\frac T{a_T}\left(\hat\theta_G-\theta\right)\xrightarrow d\ \mathcal S_\alpha,$$
$\mathcal S_\alpha$ 为一个 α-稳定分布（一般是偏斜的）。特别地，误差为 $T^{-(1-1/\alpha)}=T^{-(1-2/\nu)}$ 量级，极限分布的方差无穷。

**(b)** 若 $\alpha=2$（$\nu=4$），则 $\sqrt{T/\log T}\,(\hat\theta_G-\theta)$ 收敛到正态分布，误差为 $\sqrt{\log T/T}$ 量级。

*证明。* **第 1 步（化简）。** $Z$ 对称，故 $X$ 是椭圆分布，$\theta=g(\Sigma)$（命题 1）。由引理 A(i)，$\hat\theta_G=g(S_T)=g(M_T-\bar Y\bar Y^\top)$，
其中 $M_T=\frac1T\sum_tW_tZ_tZ_t^\top$（$\tfrac T{T-1}$ 因尺度不变而消失）。$\bar Y\bar Y^\top=O_p(T^{-1})$（$Y$ 有有限二阶矩）。

**第 2 步（线性化）。** 记 $D_t=Z_tZ_t^\top-\Sigma_0$，则
$M_T-\Sigma=\frac1T\sum_t\big[W_tD_t+(W_t-\mathbb EW)\Sigma_0\big]$。由引理 A(ii)(iii)，
$$\hat\theta_G-\theta=\frac1T\sum_t\xi_t+O_p\big(\|M_T-\Sigma\|^2\big)+O_p(T^{-1}),\qquad\xi_t=W_t\,\eta_t,\quad\eta_t=\operatorname{tr}(G_\Sigma D_t),$$
因为 $\operatorname{tr}(G_\Sigma\Sigma_0)=0$，$(W_t-\mathbb EW)\Sigma_0$ 这一项正好消失——这是比值的尺度不变性在起作用。
$\eta_t$ 与 $W_t$ 独立，$\mathbb E\eta=0$，各阶矩有限，且 $|\rho|<1$ 时不退化。

**第 3 步（Breiman 引理）。** $\mathbb E|\eta|^{\alpha+\varepsilon}<\infty$，由 Breiman 引理，
$P(\xi>x)\sim\mathbb E[\eta_+^\alpha]\,P(W>x)$，$P(\xi<-x)\sim\mathbb E[\eta_-^\alpha]\,P(W>x)$。所以 $\xi$ 的两侧尾部都是指数 α 的正则变化，均值为 0。

**第 4 步（广义中心极限定理）。** (a) $1<\alpha<2$ 时，$\frac1{a_T}\sum_t\xi_t\xrightarrow d\mathcal S_\alpha(\sigma,\beta,0)$，
偏斜参数 $\beta=(\mathbb E\eta_+^\alpha-\mathbb E\eta_-^\alpha)/(\mathbb E\eta_+^\alpha+\mathbb E\eta_-^\alpha)$（Feller 1971，第 XVII 章；Samorodnitsky & Taqqu 1994）。
对 $M_T-\Sigma$ 的每个元素同样适用（它们也是指数 α 的正则变化项之和），所以 $\|M_T-\Sigma\|=O_p(a_T/T)$。
余项 $O_p((a_T/T)^2)$ 与 $O_p(T^{-1})$ 都是 $o_p(a_T/T)$（因为 $a_T/T\to0$、$a_T\to\infty$）。乘以 $T/a_T$ 得 (a)。
(b) $\alpha=2$ 时 $\xi$ 方差无穷，但截断二阶矩 $\mathbb E[\xi^2\mathbf 1\{|\xi|\le x\}]\sim2c''\log x$ 是缓变函数，$\xi$ 属于正态分布的吸引域，规范化常数 $a_T\asymp\sqrt{T\log T}$（Feller 1971，第 XVII 章）。余项同理可忽略。$\square$

**两点说明。**

1. **为什么不能用方差检验速度。** (a) 的极限方差无穷，$\mathrm{Var}(\hat\theta_G)$ 的衰减不反映 $T^{-(1-1/\alpha)}$ 这个速度。
   分位数则可以：极限分布连续，所以 $(T/a_T)\cdot\mathrm{IQR}(\hat\theta_G)\to\mathrm{IQR}(\mathcal S_\alpha)>0$，即 IQR 按 $T^{-(1-2/\nu)}$ 衰减。
   asymptotics_check.ipynb 中 IQR 对 T 的双对数斜率：t(3) 为 −0.332（预测 −1/3），t(4) 为 −0.431（预测：$\sqrt{\log T/T}$ 在 T = 125…4000 上的平均斜率约 −0.42），
   √T 收敛的情形都在 −0.49 到 −0.51 之间。bias_variance.ipynb 第 4.3 节用方差斜率得到 t(3) −0.56、t(4) −0.75，偏离预测，原因正在于此。
2. **对早先说法的更正。** 本项目早先的讨论中曾说 ν < 4 时的极限是「正态的尺度混合」。那需要以 $W$ 为条件的 Lindeberg 条件，
   而 $W^2$ 的尾指数 $\alpha/2<1$，此时 $\max_tW_t^2/\sum_tW_t^2$ 不趋于 0，Lindeberg 条件不成立。正确的极限是上面的 α-稳定分布；收敛速度不受影响。

## 4 · 置信区间

由命题 4–6，三种 95% 区间都写成 $\hat\theta\,(1\pm1.96\sqrt{\hat V/T})$：
- 样本协方差：$\hat V_G=\tfrac14\widehat{\mathrm{Var}}(\hat u_1^2-\hat u_2^2)$，在 $\mathbb E\|X\|^4<\infty$ 下相合（强大数定律）。不需要椭圆假设，但区间覆盖的是伪真值 $\theta_G$。
- Tyler：$\hat V_{\mathrm{Ty}}=\frac{d+2}d(1-\hat\rho^2)$，$\hat\rho$ 由 $\hat V_T$ 算出；只在椭圆族中有效。
- HS：$\hat V_{HS}$ 为影响函数的样本方差，经验 VaR 代入 $v_j$。

2 < ν < 4 时，$\hat V_G$ 不收敛到有限值，Student 化统计量近似于自归一化和 $\sum\xi_t/\sqrt{\sum\xi_t^2}$；
α < 2 时它的极限不是正态分布（Logan, Mallows, Rice & Shepp 1973），所以样本协方差区间没有渐近正确的覆盖率保证。
ν = 4 时 $\xi$ 属于正态吸引域，自归一化统计量渐近正态（Giné, Götze & Mason 1997），区间渐近有效，但收敛很慢。

**覆盖率**（%，名义 95%；q = 0.95，每格 2000 次重复，蒙特卡洛标准误约 0.5 个百分点；括号内为区间宽度的中位数，相对 θ 的 %）

| DGP | T | 样本协方差 | Tyler | HS |
|---|---|---|---|---|
| 高斯 | 250 / 1000 / 4000 | 94.4 / 95.6 / 94.6 (4.2) | 94.8 / 95.2 / 94.8 (4.8) | 93.2 / 95.2 / 95.0 (7.6) |
| t(8) | 250 / 1000 / 4000 | 94.0 / 94.0 / 94.5 (5.0) | 94.6 / 94.8 / 94.2 (4.8) | 93.2 / 94.4 / 94.8 (9.4) |
| t(5) | 250 / 1000 / 4000 | 92.5 / 94.0 / 93.5 (6.2) | 94.4 / 95.6 / 95.1 (4.8) | 92.6 / 94.3 / 94.2 (10.9) |
| t(4) | 250 / 1000 / 4000 | 91.9 / 93.9 / 94.4 (7.6) | 94.6 / 95.6 / 94.2 (4.8) | 92.7 / 95.0 / 94.6 (12.1) |
| t(3) | 250 / 1000 / 4000 | 89.6 / 92.8 / 94.1 (12.1) | 94.2 / 94.6 / 95.0 (4.8) | 92.2 / 95.1 / 94.2 (14.4) |
| 崩盘混合 | 250 / 1000 / 4000 | 61.4 / 56.4 / 29.2 (12.3) | 0.4 / 0.0 / 0.0 (4.9) | 84.0 / 94.0 / 94.4 (15.8) |

（宽度取 T = 4000。）

- **Tyler 区间**在所有椭圆 DGP、所有 T 下覆盖率都在 94.2–95.6% 之间，宽度不随 ν 改变（T = 4000 时 4.8%），与命题 5 一致。
- **样本协方差区间**在 ν ≥ 8 时准确；ν 越小，小样本覆盖率越低（t(3)、T = 250 时 89.6%）。t(3) 下覆盖率随 T 回升到 94.1%，
  但上面的理论说明这不是渐近保证；而且区间宽度是 Tyler 的 2.5 倍（12.1% 对 4.8%），反映的正是命题 7 中更慢的收敛速度。
- **HS 区间**在椭圆族中 T ≥ 1000 时准确，T = 250 时略低（92–93%）；崩盘混合下它是唯一有效的区间（T ≥ 1000 时 94%）。
- **崩盘混合下**，样本协方差与 Tyler 区间都以错误的伪真值为中心，覆盖率随 T 下降（Tyler 已经为 0），这正是命题 3 所说的不相合。

## 5 · 小结

| 估计量 | 一致性 | 渐近分布（相对误差） | 条件 |
|---|---|---|---|
| 样本协方差插件 | → $\theta_G$ | $N(0,(1-\rho^2)(1+\kappa)/T)$（椭圆）；一般为 $\tfrac14\mathrm{Var}(U_1^2-U_2^2)/T$ | $\mathbb E\|X\|^4<\infty$ |
| 同上，$t_\nu$，$2<\nu<4$ | → θ | $T^{-(1-2/\nu)}$ 量级，α = ν/2 稳定极限 | 命题 7(a) |
| 同上，$t_4$ | → θ | $\sqrt{\log T/T}$ 量级，正态极限 | 命题 7(b) |
| Tyler 插件 | → $\theta_G$ | $N(0,\tfrac{d+2}d(1-\rho^2)/T)$ | 椭圆，无矩条件 |
| HS | → θ | $N(0,V_{HS}/T)$ | $\mathbb EL^2<\infty$，VaR 处密度为正 |

椭圆族中 $\theta_G=\theta$：三种估计量都相合；渐近效率的顺序由尾部厚度决定（ν < d + 4 时 Tyler 最优）。非椭圆时只有 HS 相合。

## 文献

- Acerbi, C. & Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance* 26(7).
- Chen, S. X. (2008). Nonparametric estimation of expected shortfall. *Journal of Financial Econometrics* 6(1).
- Feller, W. (1971). *An Introduction to Probability Theory and Its Applications*, Vol. II, 2nd ed. Wiley.
- Giné, E., Götze, F. & Mason, D. M. (1997). When is the Student t-statistic asymptotically standard normal? *Annals of Probability* 25(3).
- Logan, B. F., Mallows, C. L., Rice, S. O. & Shepp, L. A. (1973). Limit distributions of self-normalized sums. *Annals of Probability* 1(5).
- Muirhead, R. J. (1982). *Aspects of Multivariate Statistical Theory*. Wiley.
- Samorodnitsky, G. & Taqqu, M. S. (1994). *Stable Non-Gaussian Random Processes*. Chapman & Hall.
- Tyler, D. E. (1987). A distribution-free M-estimator of multivariate scatter. *Annals of Statistics* 15(1).
- van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge University Press.（Delta 方法、M 估计量）
