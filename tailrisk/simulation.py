"""
保留边缘分布的多资产收益模拟：核密度（KDE）边缘 + 三种注入相关的方法。

══ 研究对象 ══════════════════════════════════════════════════════

国泰君安（2018）《资产配置之步步为营：尾部风险控制与优化》（数量化专题之一百零九，
陈奥林、叶尔乐）。原文第 1.3 节：先对各资产历史收益做高斯 KDE、独立抽样得到 K；
再对历史协方差 Σ 做 Cholesky 分解 Σ = AAᵀ，用递推式注入相关：

    X_i = Σ_{j<i} (K_j − μ_j)/A_jj · A_ij + K_i

原文称「从公式的角度可以保证协方差一致」，并用逐年 KS 检验（1% 水平）验证边缘分布。两点都不成立：
    · Var(X_i) = Σ_{j<i} (A_ij/A_jj)² Var(K_j) + Var(K_i)，一般大于 Σ_ii；
    · X_i 是独立变量之和，其分布是卷积，偏度、峰度被稀释。
而且结果依赖资产的排列顺序：排在第一位的资产不受影响，越靠后的被改动越多。
（原文 3.3 节自己也承认「对相关性的捕捉方法只是近似的方法」。）

两种能严格保留边缘分布的做法，作为对照：
    gaussian_copula_kde   高斯 Copula 生成相依的均匀变量，再用各资产 KDE 的分位数函数逆变换
    iman_conover          Iman-Conover 秩重排：保留每列抽样值不变，只重排顺序以注入目标秩相关
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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


# ══════════════════════════════════════════════════════════════════
# 国泰君安（2018）的风险度量与优化
# ══════════════════════════════════════════════════════════════════
#
# 原文：X_T = log(P_T/P_0)，Y_T = min_{0≤s≤T} log(P_s/P_0)
#   期内风险 IH = P(Y_T ≤ y)
#   期末风险 EH = P(Y_T > y, X_T ≤ x)
#   总风险   TR = IH + EH = P(Y_T ≤ y 或 X_T ≤ x)
# 原文公式写作 P(X_T ≤ x, Y_T ≤ y)，但随即分解为 IH + EH，报告的数字也是两者之和，
# 实际含义是「两者之一发生」。默认 x = 1%，y = −1%，T = 20 个交易日。

def simplex_grid(k, step):
    """k 个资产、步长 step 的全部非负权重组合（和为 1）。"""
    n = int(round(1 / step))
    out = []

    def rec(prefix, left, depth):
        if depth == k - 1:
            out.append(prefix + [left])
            return
        for i in range(left + 1):
            rec(prefix + [i], left - i, depth + 1)

    rec([], n, 0)
    return np.array(out, dtype=float) / n


def path_risk(paths, W, x=0.01, y=-0.01, chunk=400):
    """
    paths：(n_paths, T, k) 模拟日简单收益；W：(m, k) 候选权重（期初配比，期内不再平衡）。
    返回 dict：TR、IH、EH、期末对数收益均值，各为长度 m 的数组。
    """
    V = np.cumprod(1 + paths, axis=1).astype(np.float32)          # 各资产的价值路径
    n, T, k = V.shape
    flat = V.reshape(n * T, k)
    out = {key: [] for key in ("TR", "IH", "EH", "mean_X")}
    for s in range(0, len(W), chunk):
        Wc = W[s:s + chunk].astype(np.float32)
        P = (flat @ Wc.T).reshape(n, T, -1)                          # (n, T, m) 组合价值
        # 比较价值而不取对数：Y ≤ y ⟺ min P ≤ e^y，X ≤ x ⟺ P_T ≤ e^x（只对期末取对数）
        PT = P[:, -1, :]
        ih = P.min(axis=1) <= np.exp(y)                              # y ≤ 0，t=0 时 P_0=1 不触发
        eh = ~ih & (PT <= np.exp(x))
        out["IH"].append(ih.mean(0)); out["EH"].append(eh.mean(0))
        out["TR"].append((ih | eh).mean(0)); out["mean_X"].append(np.log(PT).mean(0))
    return {key: np.concatenate(v) for key, v in out.items()}


def gtja_optimal_weights(paths, grid, x=0.01, y=-0.01):
    """
    在权重网格上最小化总风险 TR。TR 是路径计数的比例、对权重不连续，原文未说明求解方法；
    这里用网格穷举，并列最小时取模拟期末对数收益均值最高的一组。
    """
    r = path_risk(paths, grid, x, y)
    best = np.flatnonzero(r["TR"] == r["TR"].min())
    i = best[np.argmax(r["mean_X"][best])]
    return grid[i], {key: float(v[i]) for key, v in r.items()}


def gtja_backtest(returns, method, window=120, horizon=20, n_paths=1500, step=0.05,
                  x=0.01, y=-0.01, start=None, seed=0, order=None):
    """
    原文的滚动回测：每 horizon 个交易日，用过去 window 日的收益模拟 n_paths 条 horizon 日路径，
    在权重网格上最小化 TR，按权重配比后持有 horizon 日（期内不再平衡）。
    order：资产在模拟时的排列顺序（只影响国泰君安递推式）。
    返回 (日收益 Series, 权重 DataFrame, 模拟风险 DataFrame)。
    """
    R = returns.dropna()
    cols = list(R.columns) if order is None else order
    rng = np.random.default_rng(seed)
    grid = simplex_grid(len(cols), step)
    first = window if start is None else max(window, R.index.searchsorted(pd.Timestamp(start)))
    daily, wrec, rrec = [], {}, {}
    for t0 in range(first, len(R) - 1, horizon):
        H = R[cols].iloc[t0 - window:t0].values
        w, risk = gtja_optimal_weights(simulate_horizon(method, H, horizon, n_paths, rng), grid, x, y)
        date = R.index[t0 - 1]
        wrec[date], rrec[date] = pd.Series(w, cols), risk
        hold = R[cols].iloc[t0:t0 + horizon]
        value = (1 + hold).cumprod() @ w                              # 期初配比、期内漂移
        daily.append(value.pct_change().fillna(value.iloc[0] - 1))
    return (pd.concat(daily), pd.DataFrame(wrec).T[list(R.columns)], pd.DataFrame(rrec).T)
