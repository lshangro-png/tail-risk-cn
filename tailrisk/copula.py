"""
Copula 估计、滚动尾部相关系数、噪声基准。

══ 「λ=0」的两种判定口径 ═══════════════════════════════════════════

中金原文没有给出任何检验，只写了 Clayton 参数→0、Gumbel 参数→1 时两变量不相关。
因此「λ=0」最贴近原文的理解是**点估计落在独立边界上**。实现上，λ 是 θ 的连续
函数，θ 稍离边界时 λ 就小到可以忽略（Clayton θ=0.05 时 λ_D=2^-20≈1e-6），
所以用一个显示精度阈值 LAMBDA_ZERO_TOL 判零：

    "point"   λ < LAMBDA_ZERO_TOL 即为 0（原文口径，主口径）
    "lrtest"  另外要求独立性似然比检验显著，否则判 0（本项目加的保守口径）

══ 已知的方法局限（两种口径都有）═════════════════════════════════

Clayton/Gumbel 都只是「有正相依」的单参数模型。只要窗口内整体正相关，
MLE 就会给出离开边界的 θ，从而 λ>0 —— 即使真实的尾部相关严格为 0。
用 gaussian_null 做对照可以看到：高斯 copula（无尾部相关）在 ρ=0.3 时，
λ_D>0 的窗口占一半以上。所以 λ_D>0 的频率必须和
「相同相关强度、无尾部相关」的基准对比（expected_under_gaussian），
不能只和独立数据比。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from .marginals import rank_pit

_EPS = 1e-6
LAMBDA_ZERO_TOL = 1e-4      # 万分之一：原文 λ 以百分比两位小数展示，低于此即显示为 0.00%
ZERO_RULES = ("point", "lrtest")


def _clip01(x):
    return np.clip(np.asarray(x, dtype=float), _EPS, 1 - _EPS)


def _lr_pvalue(loglik):
    """
    独立性似然比检验 p 值。独立时两族的对数似然都恰为 0。

    θ 的独立值在参数空间边界上，严格的零分布是 χ²(0) 与 χ²(1) 的 50:50 混合；
    用纯 χ²(1) 偏保守（独立数据的实测误报率约 2%，低于名义 5%）。
    """
    return float(chi2.sf(2 * max(loglik, 0.0), df=1))


class ClaytonCopula:
    """仅下尾相关：λ_D = 2^(-1/θ)，θ→0 对应独立。"""

    name = "clayton"
    bounds = (1e-6, 30.0)

    @staticmethod
    def loglik_terms(theta, u, v):
        lu, lv = np.log(u), np.log(v)
        # log(u^-θ + v^-θ - 1) 在 log 空间计算，θ 大时 u^-θ 会溢出
        s = np.logaddexp(-theta * lu, -theta * lv)
        log_inner = s + np.log1p(-np.exp(-s))
        return np.log1p(theta) - (theta + 1) * (lu + lv) - (1 / theta + 2) * log_inner

    def fit(self, u, v):
        u, v = _clip01(u), _clip01(v)
        res = minimize_scalar(lambda t: -np.sum(self.loglik_terms(t, u, v)),
                              bounds=self.bounds, method="bounded")
        self.theta = float(res.x)
        self.loglik = float(-res.fun)
        self.pvalue = _lr_pvalue(self.loglik)
        self.lambda_raw = float(2 ** (-1 / self.theta))
        return self


class GumbelCopula:
    """仅上尾相关：λ_U = 2 - 2^(1/θ)，θ→1 对应独立。"""

    name = "gumbel"
    bounds = (1.0, 30.0)

    @staticmethod
    def loglik_terms(theta, u, v):
        x, y = -np.log(u), -np.log(v)
        lx, ly = np.log(x), np.log(y)
        log_A = np.logaddexp(theta * lx, theta * ly)
        A_inv = np.exp(log_A / theta)
        return (-A_inv + x + y + (theta - 1) * (lx + ly)
                + (1 / theta - 2) * log_A + np.log(theta - 1 + A_inv))

    def fit(self, u, v):
        u, v = _clip01(u), _clip01(v)
        res = minimize_scalar(lambda t: -np.sum(self.loglik_terms(t, u, v)),
                              bounds=self.bounds, method="bounded")
        self.theta = float(res.x)
        self.loglik = float(-res.fun)
        self.pvalue = _lr_pvalue(self.loglik)
        self.lambda_raw = float(2 - 2 ** (1 / self.theta))
        return self


class GaussianCopula:
    """基准模型，尾部相关系数恒为 0 —— 这正是它低估尾部风险的原因。"""

    name = "gaussian"

    def fit(self, u, v):
        u, v = _clip01(u), _clip01(v)
        x, y = norm.ppf(u), norm.ppf(v)
        self.rho = float(np.corrcoef(x, y)[0, 1])
        det = 1 - self.rho ** 2
        self.loglik = float(np.sum(
            -0.5 * np.log(det)
            - (self.rho ** 2 * (x ** 2 + y ** 2) - 2 * self.rho * x * y) / (2 * det)))
        return self


def zero_out(lambda_raw, pvalue, rule="point", alpha=0.05, tol=LAMBDA_ZERO_TOL):
    """按判零口径把 λ 原始估计置零。标量或数组均可。"""
    lam = np.asarray(lambda_raw, dtype=float)
    is_zero = lam < tol
    if rule == "lrtest":
        is_zero = is_zero | (np.asarray(pvalue, dtype=float) > alpha)
    elif rule != "point":
        raise ValueError(f"zero_rule 须为 {ZERO_RULES}")
    return np.where(is_zero, 0.0, lam)


# ══════════════════════════════════════════════════════════════════
# 单窗口拟合与滚动估计
# ══════════════════════════════════════════════════════════════════

def fit_window(a, b, pit="window"):
    """
    对一个窗口拟合 Clayton + Gumbel，返回原始结果（未判零）。

    pit="window"：a, b 是收益率，在窗口内做秩变换（主口径）
    pit="given"： a, b 已是 (0,1) 上的 PIT 值，直接使用
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    u, v = (rank_pit(a), rank_pit(b)) if pit == "window" else (a, b)
    c, g = ClaytonCopula().fit(u, v), GumbelCopula().fit(u, v)
    return {
        "lambda_D_raw": c.lambda_raw, "lambda_U_raw": g.lambda_raw,
        "p_D": c.pvalue, "p_U": g.pvalue,
        "clayton_theta": c.theta, "gumbel_theta": g.theta,
        "clayton_ll": c.loglik, "gumbel_ll": g.loglik,
        "ll_diff": c.loglik - g.loglik,
        # 正态得分相关系数：窗口内「整体相关强度」，用于匹配高斯基准
        "ns_corr": float(np.corrcoef(norm.ppf(_clip01(u)), norm.ppf(_clip01(v)))[0, 1]),
    }


def apply_zero_rule(roll, rule="point", alpha=0.05, tol=LAMBDA_ZERO_TOL):
    """按判零口径生成 lambda_D / lambda_U 列。不重新拟合，切换口径很便宜。"""
    out = roll.copy()
    out["lambda_D"] = zero_out(out.lambda_D_raw, out.p_D, rule, alpha, tol)
    out["lambda_U"] = zero_out(out.lambda_U_raw, out.p_U, rule, alpha, tol)
    return out


def rolling_tail_dependence(x, y, window=60, step=1, pit="window",
                            zero_rule="point", alpha=0.05):
    """
    滚动窗口内分别拟合 Clayton 与 Gumbel，各取 λ_D / λ_U。
    中金原文口径：60 日窗口、每日滚动、两者独立估计、不做模型选择。

    x, y：pit="window" 时传收益率；pit="given" 时传已变换好的 PIT 值。
    返回 DataFrame（index=窗口最后一天），含原始估计与按 zero_rule 判零后的 λ。
    """
    df = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    xs, ys, idx = df.iloc[:, 0].values, df.iloc[:, 1].values, df.index

    rows, dates = [], []
    for end in range(window, len(idx) + 1, step):
        rows.append(fit_window(xs[end - window:end], ys[end - window:end], pit))
        dates.append(idx[end - 1])
    roll = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
    return apply_zero_rule(roll, zero_rule, alpha)


# ══════════════════════════════════════════════════════════════════
# 噪声基准
# ══════════════════════════════════════════════════════════════════

def gaussian_null(window=60, n_sims=4000, rho=None, seed=1):
    """
    在高斯 copula（尾部相关严格为 0）下模拟窗口，走与实测完全相同的
    「窗口内秩变换 → 拟合」流程，返回每次模拟的原始结果。

    rho=None   ρ 在 [-0.9, 0.9] 上均匀抽取，用于构造「给定窗口相关强度下的误报率曲线」
    rho=数值   固定相关系数
    rho=数组   每次从中有放回抽取（如传入实测的 ns_corr，使基准的相关分布与实测一致）
    """
    rng = np.random.default_rng(seed)
    pool = None if rho is None or np.isscalar(rho) else np.clip(np.asarray(rho, float), -0.95, 0.95)
    rows = []
    for _ in range(n_sims):
        if rho is None:
            r = rng.uniform(-0.9, 0.9)
        elif pool is not None:
            r = rng.choice(pool)
        else:
            r = rho
        z1 = rng.standard_normal(window)
        z2 = r * z1 + np.sqrt(1 - r ** 2) * rng.standard_normal(window)
        res = fit_window(z1, z2, pit="window")
        res["rho_true"] = r
        rows.append(res)
    return pd.DataFrame(rows)


def noise_baseline(window=60, n_rep=300, rho=0.0, seed=1, zero_rule="point", alpha=0.05):
    """固定 ρ 的高斯基准下，三个判别量的命中率。ρ=0 即纯独立数据。"""
    sim = apply_zero_rule(gaussian_null(window, n_rep, rho, seed), zero_rule, alpha)
    ld, lu = sim.lambda_D > 0, sim.lambda_U > 0
    return {
        "rho": rho, "判零口径": zero_rule,
        "lambda_D>0占比": ld.mean(),
        "lambda_U>0占比": lu.mean(),
        "中金条件命中率": (ld & ~lu).mean(),
    }


def expected_under_gaussian(roll, null, zero_rule="point", alpha=0.05, bin_width=0.05):
    """
    对实测的每个窗口，按其正态得分相关系数 ns_corr 查高斯基准下的命中概率。

    返回 DataFrame（与 roll 同 index），三列分别是在
    「相同整体相关强度、但无尾部相关」的世界里，该窗口出现
    λ_D>0 / λ_U>0 / 中金条件 的概率。对时间取平均即为匹配基准。
    """
    sim = apply_zero_rule(null, zero_rule, alpha)
    edges = np.arange(-1.0, 1.0 + bin_width, bin_width)
    sim_bin = np.digitize(sim.ns_corr, edges)
    table = pd.DataFrame({
        "lambda_D>0": sim.lambda_D > 0,
        "lambda_U>0": sim.lambda_U > 0,
        "中金条件": (sim.lambda_D > 0) & (sim.lambda_U == 0),
    }).groupby(sim_bin).mean()
    # 空箱用相邻箱插值
    table = table.reindex(range(len(edges) + 1)).interpolate(limit_direction="both")
    return table.reindex(np.digitize(roll.ns_corr, edges)).set_axis(roll.index)
