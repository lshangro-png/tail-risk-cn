"""
椭圆族 Copula：高斯与 t。用来回答「有没有超出普通相关的尾部联动」。

══ 为什么用 t 对比高斯 ═══════════════════════════════════════════════

t-Copula 有两个参数，分工明确：
    ρ  整体相关强度（与高斯 Copula 的 ρ 含义相同）
    ν  自由度，管尾部厚度：ν 越小尾部越厚，ν→∞ 退化为高斯（尾部相关为 0）

所以「t 的似然是否显著好于高斯」恰好检验：在同样的整体相关下，
极端值是否比高斯所暗示的更容易同时出现。这是 Clayton 单参数做不到的。

══ 解读时的两个注意 ══════════════════════════════════════════════

1. t 的尾部相关系数 λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) 也随 ρ 增大，
   不能只看 λ 的数值，要看 ν 是否显著有限（即 LR 检验）。
2. t 的上下尾对称，且 ρ≈0 时四个角都会变厚。共同的波动率冲击
   （两个市场同一天都大幅波动、方向随机）也会让 ν 变小。
   所以 ν 显著有限说明「极端值倾向于同日出现」，不等于「倾向于同日下跌」。
   方向性要用 SJC 等非对称模型另行检验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2, kendalltau, norm
from scipy.stats import t as student_t

from .marginals import rank_pit

NU_MAX = 200.0       # ν 上限：超过即视同高斯
NU_MIN = 2.05


def _clip01(x, eps=1e-6):
    return np.clip(np.asarray(x, dtype=float), eps, 1 - eps)


# ══════════════════════════════════════════════════════════════════
# 对数似然
# ══════════════════════════════════════════════════════════════════

def gaussian_loglik(rho, u, v):
    x, y = norm.ppf(u), norm.ppf(v)
    det = 1 - rho ** 2
    return np.sum(-0.5 * np.log(det) - (rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * det))


def t_loglik(rho, nu, u, v):
    x, y = student_t.ppf(u, nu), student_t.ppf(v, nu)
    det = 1 - rho ** 2
    q = (x ** 2 - 2 * rho * x * y + y ** 2) / (nu * det)
    log_joint = (gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi)
                 - 0.5 * np.log(det) - (nu + 2) / 2 * np.log1p(q))
    # 两个一元 t 密度之积（归一化常数计两次）
    log_marg = (2 * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi))
                - (nu + 1) / 2 * (np.log1p(x ** 2 / nu) + np.log1p(y ** 2 / nu)))
    return np.sum(log_joint - log_marg)


def t_tail_dependence(rho, nu):
    """t-Copula 的尾部相关系数（上下尾相同）。"""
    if not np.isfinite(nu) or nu >= NU_MAX:
        return 0.0
    return float(2 * student_t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1))


# ══════════════════════════════════════════════════════════════════
# 估计与检验
# ══════════════════════════════════════════════════════════════════

def fit_gaussian(u, v):
    u, v = _clip01(u), _clip01(v)
    res = minimize_scalar(lambda r: -gaussian_loglik(r, u, v), bounds=(-0.99, 0.99), method="bounded")
    return float(res.x), float(-res.fun)


def fit_t(u, v):
    """
    极大似然估计 (ρ, ν)。ρ 的初值取 Kendall τ 的椭圆族换算 sin(πτ/2)，
    ν 从厚尾、薄尾两个初值出发，取似然更高者，避免停在局部解。
    """
    u, v = _clip01(u), _clip01(v)
    tau = kendalltau(u, v)[0]
    rho0 = float(np.clip(np.sin(np.pi * tau / 2), -0.95, 0.95))
    best = None
    for nu0 in (6.0, 60.0):
        res = minimize(lambda p: -t_loglik(p[0], p[1], u, v), x0=[rho0, nu0],
                       bounds=[(-0.99, 0.99), (NU_MIN, NU_MAX)], method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    rho, nu = map(float, best.x)
    return rho, nu, float(-best.fun)


def t_vs_gaussian(u, v):
    """
    似然比检验 H0：高斯（ν=∞） vs H1：t（ν 有限）。

    ν=∞ 在参数空间边界上，LR 的零分布是 ½χ²(0)+½χ²(1)，
    故 p = ½·P(χ²(1) ≥ LR)。test_t_lr_size_under_gaussian 检验了这一校准。
    """
    rho_g, ll_g = fit_gaussian(u, v)
    rho_t, nu, ll_t = fit_t(u, v)
    lr = max(2 * (ll_t - ll_g), 0.0)
    p = 0.5 * chi2.sf(lr, 1) if lr > 0 else 1.0
    return {"rho": rho_g, "rho_t": rho_t, "nu": nu, "lr": lr, "p_t": p,
            "lambda_t": t_tail_dependence(rho_t, nu)}


def rolling_t_test(x, y, window=250, step=1):
    """
    滚动窗口内（窗口内秩变换）做 t vs 高斯检验。
    x, y 为收益率；返回 DataFrame（index = 窗口最后一天）。
    """
    df = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    xs, ys, idx = df.iloc[:, 0].values, df.iloc[:, 1].values, df.index
    rows, dates = [], []
    for end in range(window, len(idx) + 1, step):
        a, b = xs[end - window:end], ys[end - window:end]
        rows.append(t_vs_gaussian(rank_pit(a), rank_pit(b)))
        dates.append(idx[end - 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def empirical_tail_coexceedance(x, y, q=0.10):
    """
    非参数：股票落在最差 q 分位的日子里，债券也落在最差 q 分位的比例（下尾），
    以及对应的上尾比例。用于不依赖模型的交叉验证；高斯/独立时约等于 q 附近。
    """
    u, v = rank_pit(np.asarray(x)), rank_pit(np.asarray(y))
    lo, hi = u <= q, u > 1 - q
    return {"下尾同落": float((v[lo] <= q).mean()), "上尾同落": float((v[hi] > 1 - q).mean())}


def gaussian_null_t_test(window=250, n_sims=500, rho=0.0, seed=1):
    """高斯数据下 t-vs-高斯 检验的拒绝率（检验的名义水平校准用）。"""
    rng = np.random.default_rng(seed)
    pool = None if np.isscalar(rho) else np.asarray(rho, float)
    out = []
    for _ in range(n_sims):
        r = rng.choice(pool) if pool is not None else rho
        z1 = rng.standard_normal(window)
        z2 = r * z1 + np.sqrt(1 - r ** 2) * rng.standard_normal(window)
        out.append(t_vs_gaussian(rank_pit(z1), rank_pit(z2)))
    return pd.DataFrame(out)
