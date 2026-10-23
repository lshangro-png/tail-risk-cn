"""
因子拥挤度与 alpha 衰减 —— Lee（2025）《Not All Factors Crowd Equally》（arXiv 2512.11913）。

══ 原文 v1（2025-12）的做法 ══════════════════════════════════════

    alpha 代理：滚动 36 个月夏普比率
    衰减模型：双曲线 α(t) = K/(1+λt)，与指数 K·e^(−λt)、线性 K − bt 对比 R²
    拥挤信号：在扩展窗口（至少 120 个月）上拟合双曲线，残差 = 实际夏普 − 预测夏普，负残差 = 拥挤
    尾部检验：1980–2000 年定阈值（残差中位数为拥挤分界，因子收益 10% 分位为崩盘线），
              2001–2024 年样本外比较拥挤 / 不拥挤时的崩盘概率

v1 未说明：时间 t 的起点、拟合方法、崩盘看当月还是下月收益、p 值所用的检验。

══ 时间错位（作者在 v3 中自己指出）══════════════════════════════

月份 t 的滚动夏普包含 t 月收益。t 月大跌 → 滚动夏普下降 → 残差变负 → 被判为「拥挤」。
若崩盘也看 t 月收益，「拥挤时崩盘更多」就只是同一个事件被数了两次，不是预测。
本模块对崩盘同时提供当月（horizon=0）与下月（horizon=1）两种口径，并在纯随机数据上走同一流程作对照。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import fisher_exact

DECAY = {
    "双曲线": (lambda t, K, lam: K / (1 + lam * t), [0.5, 0.1], ([-np.inf, 0], [np.inf, np.inf])),
    "指数": (lambda t, K, lam: K * np.exp(-lam * t), [0.5, 0.05], ([-np.inf, 0], [np.inf, np.inf])),
    "线性": (lambda t, K, b: K - b * t, [0.5, 0.01], ([-np.inf, -np.inf], [np.inf, np.inf])),
}


def rolling_sharpe(r, window=36):
    """年化滚动夏普：均值 / 标准差 × √12。t 月的值包含 t 月收益。"""
    r = pd.Series(r).dropna()
    return (r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(12)).dropna()


def years_since(index, origin):
    return np.asarray((pd.DatetimeIndex(index) - pd.Timestamp(origin)).days / 365.25, float)


def fit_decay(t, y, kind="双曲线"):
    """非线性最小二乘拟合衰减曲线，返回 (参数, R²)。拟合失败返回 (None, nan)。"""
    f, p0, bounds = DECAY[kind]
    t, y = np.asarray(t, float), np.asarray(y, float)
    try:
        p, _ = curve_fit(f, t, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None, np.nan
    resid = y - f(t, *p)
    return p, float(1 - resid.var() / y.var()) if y.var() > 0 else np.nan


def decay_r2(sharpe, start="1995-01-01", end="2024-12-31", origin="1995-01-01", positive_only=True):
    """v1 表 2：在 [start, end] 的滚动夏普上分别拟合三种衰减曲线，返回 R²。"""
    s = sharpe.loc[start:end]
    if positive_only:
        s = s[s > 0]
    t = years_since(s.index, origin)
    return {k: fit_decay(t, s.values, k)[1] for k in DECAY}


def crowding_residual(sharpe, min_obs=120, kind="双曲线"):
    """
    v1 的拥挤信号：在 t 月，用严格早于 t 的全部滚动夏普（至少 min_obs 个）拟合衰减曲线，
    预测 t 月的值；残差 = 实际 − 预测（负 = 拥挤）。时间以第一个完整窗口为零点（年）。
    """
    s = sharpe.dropna()
    t_all = years_since(s.index, s.index[0])
    f = DECAY[kind][0]
    out = {}
    for i in range(min_obs, len(s)):
        p, _ = fit_decay(t_all[:i], s.values[:i], kind)
        if p is not None:
            out[s.index[i]] = s.values[i] - f(t_all[i], *p)
    return pd.Series(out)


def crash_table(returns, signal, horizon=0, train=("1980-01-01", "2000-12-31"), test=("2001-01-01", "2024-12-31"),
                crowded_below=True):
    """
    v1 表 4 的结构：
        拥挤 = signal 低于训练期中位数（crowded_below=False 时为高于）
        崩盘 = t+horizon 月收益低于训练期收益的 10% 分位
    在测试期统计 P(崩盘|拥挤)、P(崩盘|不拥挤)、比值，以及 Fisher 精确检验（双侧）p 值。
    """
    r = pd.Series(returns).dropna()
    sig = pd.Series(signal).dropna()
    med = sig.loc[train[0]:train[1]].median()
    thr = r.loc[train[0]:train[1]].quantile(0.10)
    crash = (r < thr).astype(int).shift(-horizon)
    crowded = (sig < med) if crowded_below else (sig > med)
    df = pd.concat({"拥挤": crowded, "崩盘": crash}, axis=1).loc[test[0]:test[1]].dropna()
    is_c = df["拥挤"].astype(bool)                               # 拼接后为 object 类型，取反前先转成布尔
    c, u = df.loc[is_c, "崩盘"], df.loc[~is_c, "崩盘"]
    table = [[int(c.sum()), int(len(c) - c.sum())], [int(u.sum()), int(len(u) - u.sum())]]
    p_c, p_u = c.mean(), u.mean()
    return {"P(崩盘|拥挤)": p_c, "P(崩盘|不拥挤)": p_u, "比值": p_c / p_u if p_u > 0 else np.nan,
            "Fisher p": fisher_exact(table)[1], "拥挤月数": len(c), "不拥挤月数": len(u)}


def null_crash_ratios(mu, sigma, n_months, n_sims=200, window=36, min_obs=120, seed=0, horizons=(0, 1),
                      train_frac=(16 / 61, 37 / 61), test_frac=(38 / 61, 61 / 61)):
    """
    纯随机对照：收益独立同分布 N(mu, sigma²)，没有任何拥挤或衰减。
    走与真实数据完全相同的流程（滚动夏普 → 扩展窗口双曲线残差 → 训练期定阈值 → 测试期统计），
    返回每次模拟、每个 horizon 的「拥挤 / 不拥挤」崩盘概率比值。
    训练、测试期按真实数据的相对位置切分。
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1963-07-31", periods=n_months, freq="ME")
    tr = (idx[int(train_frac[0] * n_months)], idx[int(train_frac[1] * n_months) - 1])
    te = (idx[int(test_frac[0] * n_months)], idx[int(test_frac[1] * n_months) - 1])
    rows = []
    for _ in range(n_sims):
        r = pd.Series(rng.normal(mu, sigma, n_months), idx)
        res = crowding_residual(rolling_sharpe(r, window), min_obs)
        rows.append({h: crash_table(r, res, h, train=tr, test=te)["比值"] for h in horizons})
    return pd.DataFrame(rows)
