"""
对称化 Joe-Clayton（SJC）Copula —— Patton (2006, International Economic Review)。

══ 为什么用 SJC ═══════════════════════════════════════════════════

中金分别拟合 Clayton（只有下尾）和 Gumbel（只有上尾），两个模型各估各的，
无法回答「下尾是否强于上尾」。SJC 在同一个模型里有两个参数，
直接就是两个尾部相关系数：

    τ_L = 下尾相关系数，τ_U = 上尾相关系数

「下尾强于上尾」因此可以写成一个正式的似然比检验：
    H0: τ_L = τ_U（对称）  vs  H1: τ_L ≠ τ_U，  LR ~ χ²(1)

⚠ 与 Clayton 一样，SJC 只能描述正相依；整体相关越强，τ_L、τ_U 都越大。
所以 τ 的绝对水平仍然混有普通相关，**看差值（不对称检验）比看水平可靠**。
负相关时两者都落到 0，此时不对称检验无意义（返回 NaN）。

══ 旋转：看「股跌债涨」 ═══════════════════════════════════════════

把第二个变量翻转（v → 1−v）后再拟合，原来的「左上角」变成「左下角」：
    τ_L(u, 1−v) = 股票极端下跌时，债券同时极端上涨的概率（避险 / flight-to-quality）
    τ_U(u, 1−v) = 股票极端上涨时，债券同时极端下跌的概率
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2

TAU_MIN, TAU_MAX = 1e-4, 0.95     # 上限 0.95：再大 κ 会大到 (1−u)^κ 下溢；股债尾部相关远小于此


def _clip01(x, eps=1e-6):
    return np.clip(np.asarray(x, dtype=float), eps, 1 - eps)


def _log1mexp(a):
    """log(1 − e^a)，a<0。a 接近 0 时用 expm1、远离 0 时用 log1p，两端都不丢精度。"""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > -0.6931, np.log(-np.expm1(a)), np.log1p(-np.exp(a)))


def jc_logpdf(u, v, tau_u, tau_l):
    """
    Joe-Clayton（BB7）copula 的对数密度。κ = 1/log2(2−τ_U)，γ = −1/log2(τ_L)。

    记 x = 1−(1−u)^κ，y = 1−(1−v)^κ，A = x^−γ + y^−γ − 1，h = A^(−1/γ)，w = 1−h，
    C = 1 − w^(1/κ)。对 u、v 求二阶混合偏导后整理得：
        c = a(u)·b(v)/κ · w^(1/κ−2) · A^(−1/γ−2) · [(1−1/κ)·h + (1+γ)·w]
    其中 a(u) = κ(1−u)^(κ−1)·x^(−γ−1)，b(v) 同理。全部在对数空间计算。
    """
    k = 1.0 / np.log2(2.0 - tau_u)
    g = -1.0 / np.log2(tau_l)
    lu, lv = np.log1p(-u), np.log1p(-v)                  # log(1−u), log(1−v)
    lx = _log1mexp(k * lu)                               # log x，x = 1 − (1−u)^κ
    ly = _log1mexp(k * lv)
    # log A，A = x^−γ + y^−γ − 1。右上角（u,v→1）时 A 只比 1 大一点点，
    # 直接相减会丢精度让 w=0、似然变 +inf，所以用 log1p(expm1+expm1)；
    # 左下角 A 极大时 expm1 会溢出，改用 logaddexp。两者按大小切换。
    ex, ey = -g * lx, -g * ly
    with np.errstate(over="ignore", invalid="ignore"):
        near_one = np.log1p(np.expm1(ex) + np.expm1(ey))
        s = np.logaddexp(ex, ey)
        far = s + np.log1p(-np.exp(-s))
    log_A = np.where(np.maximum(ex, ey) < 30, near_one, far)
    log_h = -log_A / g
    log_w = _log1mexp(log_h)
    h, w = np.exp(log_h), np.exp(log_w)
    log_a = np.log(k) + (k - 1) * lu + (-g - 1) * lx
    log_b = np.log(k) + (k - 1) * lv + (-g - 1) * ly
    return (log_a + log_b - np.log(k) + (1 / k - 2) * log_w + (-1 / g - 2) * log_A
            + np.log((1 - 1 / k) * h + (1 + g) * w))


def sjc_logpdf(u, v, tau_u, tau_l):
    """SJC = ½[JC(u,v; τU,τL) + JC(1−u,1−v; τL,τU)]，上下尾相关系数恰为 τ_U、τ_L。"""
    return np.log(0.5) + np.logaddexp(jc_logpdf(u, v, tau_u, tau_l),
                                      jc_logpdf(1 - u, 1 - v, tau_l, tau_u))


def fit_sjc(u, v, symmetric=False):
    """极大似然估计。symmetric=True 时约束 τ_L = τ_U（不对称检验的原假设）。"""
    u, v = _clip01(u), _clip01(v)
    if symmetric:
        res = minimize_scalar(lambda t: -np.sum(sjc_logpdf(u, v, t, t)),
                              bounds=(TAU_MIN, TAU_MAX), method="bounded")
        return float(res.x), float(res.x), float(-res.fun)
    best = None
    for start in ([0.1, 0.1], [0.3, 0.05], [0.05, 0.3]):
        res = minimize(lambda p: -np.sum(sjc_logpdf(u, v, p[0], p[1])), x0=start,
                       bounds=[(TAU_MIN, TAU_MAX)] * 2, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    tau_u, tau_l = map(float, best.x)
    return tau_u, tau_l, float(-best.fun)


def asymmetry_test(u, v, min_tau=0.01):
    """
    检验 H0: τ_L = τ_U。返回两尾估计、LR 统计量与 p 值。
    对称约束下的估计也落在 0 附近时（整体负相关或不相关），没有尾部可比，p 值记为 NaN。
    """
    tau_u, tau_l, ll = fit_sjc(u, v)
    tau_s, _, ll_sym = fit_sjc(u, v, symmetric=True)
    lr = max(2 * (ll - ll_sym), 0.0)
    valid = max(tau_u, tau_l) >= min_tau
    return {"tau_L": tau_l, "tau_U": tau_u, "tau_sym": tau_s, "lr": lr,
            "p_asym": float(chi2.sf(lr, 1)) if valid else np.nan}


def rolling_asymmetry(u_returns, v_returns, window=250, step=5, flip_v=False):
    """滚动窗口（窗口内秩变换）的 SJC 不对称检验。flip_v=True 时检验旋转后的 (u, 1−v)。"""
    from .marginals import rank_pit

    df = pd.concat([pd.Series(u_returns), pd.Series(v_returns)], axis=1).dropna()
    xs, ys, idx = df.iloc[:, 0].values, df.iloc[:, 1].values, df.index
    rows, dates = [], []
    for end in range(window, len(idx) + 1, step):
        u, v = rank_pit(xs[end - window:end]), rank_pit(ys[end - window:end])
        rows.append(asymmetry_test(u, 1 - v if flip_v else v))
        dates.append(idx[end - 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def gaussian_tau_baseline(rhos, window=250, n_sims=100, seed=1):
    """
    高斯 Copula（尾部相关为 0、上下尾对称）下，SJC 估出的 τ 平均有多大。

    SJC 的 τ 水平混有普通相关：ρ=0.3 的高斯数据也会估出 τ≈0.12。
    用这张表扣掉「仅由相关强度带来的 τ」，剩下的才是超出普通相关的尾部联动。
    返回 DataFrame，index=ρ，列为 τ_L、τ_U 的均值及不对称检验 5% 水平的拒绝率。
    """
    from .marginals import rank_pit

    rng = np.random.default_rng(seed)
    rows = {}
    for rho in rhos:
        res = []
        for _ in range(n_sims):
            z1 = rng.standard_normal(window)
            z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(window)
            res.append(asymmetry_test(rank_pit(z1), rank_pit(z2)))
        res = pd.DataFrame(res)
        rows[rho] = {"tau_L均值": res.tau_L.mean(), "tau_U均值": res.tau_U.mean(),
                     "不对称检验误报率": (res.p_asym < 0.05).mean()}
    return pd.DataFrame(rows).T.rename_axis("rho")
