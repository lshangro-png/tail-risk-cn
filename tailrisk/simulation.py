"""
保留边缘分布的多资产收益模拟：核密度（KDE）边缘 + 三种注入相关的方法。

══ 研究对象 ══════════════════════════════════════════════════════

国泰君安（2018）《资产配置之步步为营：尾部风险控制与优化》（数量化专题之一百零九）
据整理稿记载：先对各资产历史收益做高斯 KDE、独立抽样得到 K；再对历史协方差 Σ 做
Cholesky 分解 Σ = AAᵀ，用递推式注入相关：

    X_i = Σ_{j<i} (K_j − μ_j)/A_jj · A_ij + K_i

并称该式同时保证 Cov(X) = Σ、且 X_i 的边缘分布与原始资产一致。两点都不成立：
    · Var(X_i) = Σ_{j<i} (A_ij/A_jj)² Var(K_j) + Var(K_i)，一般大于 Σ_ii；
    · X_i 是独立变量之和，其分布是卷积，偏度、峰度被稀释。
而且结果依赖资产的排列顺序：排在第一位的资产不受影响，越靠后的被改动越多。
（未能取得原文全文，递推式取自整理稿，无法核对是否抄写准确。）

两种能严格保留边缘分布的做法，作为对照：
    gaussian_copula_kde   高斯 Copula 生成相依的均匀变量，再用各资产 KDE 的分位数函数逆变换
    iman_conover          Iman-Conover 秩重排：保留每列抽样值不变，只重排顺序以注入目标秩相关
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


# ══════════════════════════════════════════════════════════════════
# 高斯 KDE 的抽样与分位数
# ══════════════════════════════════════════════════════════════════

def kde_bandwidth(x):
    """Silverman 经验带宽。"""
    x = np.asarray(x, float)
    s = min(x.std(ddof=1), (np.quantile(x, .75) - np.quantile(x, .25)) / 1.349)
    return 0.9 * s * len(x) ** (-0.2)


def kde_sample(x, n, rng):
    """从高斯 KDE 精确抽样：随机取一个历史点，再加上 N(0, h²) 噪声。"""
    x = np.asarray(x, float)
    return rng.choice(x, n) + kde_bandwidth(x) * rng.standard_normal(n)


def kde_ppf(x, u, grid_size=4000):
    """高斯 KDE 的分位数函数（在细网格上数值求逆 CDF）。"""
    x = np.asarray(x, float)
    h = kde_bandwidth(x)
    grid = np.linspace(x.min() - 6 * h, x.max() + 6 * h, grid_size)
    cdf = norm.cdf((grid[:, None] - x[None, :]) / h).mean(axis=1)
    return np.interp(u, cdf, grid)


# ══════════════════════════════════════════════════════════════════
# 三种注入相关的方法（输入 hist：T×k 历史收益；输出 n×k 模拟收益）
# ══════════════════════════════════════════════════════════════════

def gtja_recursion(hist, n, rng):
    """整理稿记载的国泰君安递推式。"""
    H = np.asarray(hist, float)
    k = H.shape[1]
    K = np.column_stack([kde_sample(H[:, j], n, rng) for j in range(k)])
    A = np.linalg.cholesky(np.cov(H, rowvar=False))
    mu = H.mean(axis=0)
    X = K.copy()
    for i in range(1, k):
        for j in range(i):
            X[:, i] += (K[:, j] - mu[j]) / A[j, j] * A[i, j]
    return X


def gaussian_copula_kde(hist, n, rng):
    """正态得分相关 → 多元正态 → Φ 得到相依的均匀变量 → 各资产 KDE 分位数逆变换。"""
    H = np.asarray(hist, float)
    T, k = H.shape
    Zs = norm.ppf(np.column_stack([rankdata(H[:, j]) / (T + 1) for j in range(k)]))
    C = np.corrcoef(Zs, rowvar=False)
    U = norm.cdf(rng.multivariate_normal(np.zeros(k), C, n))
    return np.column_stack([kde_ppf(H[:, j], U[:, j]) for j in range(k)])


def iman_conover(hist, n, rng):
    """
    Iman & Conover (1982)：各列独立地从 KDE 抽样，再按目标秩相关重排各列顺序。
    目标相关取历史收益的正态得分相关，与 gaussian_copula_kde 一致。
    """
    H = np.asarray(hist, float)
    T, k = H.shape
    K = np.column_stack([np.sort(kde_sample(H[:, j], n, rng)) for j in range(k)])
    Zs = norm.ppf(np.column_stack([rankdata(H[:, j]) / (T + 1) for j in range(k)]))
    C = np.corrcoef(Zs, rowvar=False)
    # 得分矩阵：van der Waerden 得分的随机排列，去相关后再乘目标相关的 Cholesky 因子
    S = np.column_stack([rng.permutation(norm.ppf(np.arange(1, n + 1) / (n + 1))) for _ in range(k)])
    E = np.corrcoef(S, rowvar=False)
    S_star = S @ np.linalg.inv(np.linalg.cholesky(E)).T @ np.linalg.cholesky(C).T
    out = np.empty_like(K)
    for j in range(k):
        out[:, j] = K[:, j][rankdata(S_star[:, j], method="ordinal").astype(int) - 1]
    return out


METHODS = {"国泰君安递推": gtja_recursion, "高斯Copula+KDE": gaussian_copula_kde, "Iman-Conover": iman_conover}


def simulate_horizon(method, hist, horizon, n_paths, rng):
    """
    模拟 n_paths 条长度为 horizon 的日收益路径（各日独立同分布），返回 (n_paths, horizon, k)。
    """
    k = np.asarray(hist).shape[1]
    draws = method(hist, n_paths * horizon, rng)
    return draws.reshape(n_paths, horizon, k)
