"""
单资产尾部风险预警 —— 兴业证券（2025）《基于宏观相似、动量趋势和尾部风险控制的全球资产配置研究》第 4 章。

══ 原文的信号 ═══════════════════════════════════════════════════════
（「超过过去 N 的 99% 分位数」「近五日斜率为正」以下分别记为 exceeds、rising）

    沪深300/中证500  股债性价比（全A股息率 − 10年国债收益率）低于过去 4 年的 10% 分位
                    且 指数近 5 日累计收益为负 且 PcaScore 近 21 日均值 > 近 63 日均值
    恒生指数         美元兑离岸人民币 exceeds(2 年) 且 美国 10 年实际利率 exceeds(1 年) 且 rising
    标普500         VIX exceeds(1 年) 且 rising  或  实际利率 exceeds(2 年) 且 rising
    纳斯达克100      同上，VIX 换成 VXN
    黄金            实际利率 exceeds(3 个月) 且 rising
    原油            OVX exceeds(1 年) 且 rising
    南华商品指数      美元指数 exceeds(2 年) 且 rising（免费数据不足，未复现）

触发后：事件研究看未来 1/2/3/5/10 日平均收益；择时策略在信号后 5 个交易日空仓。

══ 实现上的取法（原文未说明）════════════════════════════════════════

    · 分位数阈值只用截至前一日的数据（shift(1)），当日值与之比较
    · 「近五日斜率」取最近 5 个观测值线性回归的斜率
    · 1 年 / 2 年 / 3 个月 / 4 年 = 252 / 504 / 63 / 1008 个观测
    · PcaScore：窗口内各成分收益协方差矩阵前 2 个特征值之和 / 前 20 个之和
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR = 252


# ══════════════════════════════════════════════════════════════════
# 基础构件
# ══════════════════════════════════════════════════════════════════

def exceeds(x, window, q=0.99):
    """当日值是否高于截至前一日、过去 window 个观测的 q 分位数。"""
    x = pd.Series(x).dropna()
    thr = x.rolling(window, min_periods=window).quantile(q).shift(1)
    return (x > thr) & thr.notna()


def below(x, window, q=0.10):
    """当日值是否低于截至前一日、过去 window 个观测的 q 分位数。"""
    x = pd.Series(x).dropna()
    thr = x.rolling(window, min_periods=window).quantile(q).shift(1)
    return (x < thr) & thr.notna()


def rising(x, n=5):
    """最近 n 个观测的线性回归斜率是否为正。"""
    x = pd.Series(x).dropna()
    t = np.arange(n) - (n - 1) / 2
    slope = x.rolling(n).apply(lambda v: np.dot(t, v - v.mean()), raw=True)
    return slope > 0


def pca_score(returns, window=21, top=2, total=20):
    """
    PcaScore：窗口内收益协方差矩阵的前 top 个特征值之和，占前 total 个之和的比例。
    数值越高，成分之间走势越趋同。
    """
    R = returns.dropna()
    X = R.values
    out = np.full(len(R), np.nan)
    for end in range(window, len(R) + 1):
        ev = np.sort(np.linalg.eigvalsh(np.cov(X[end - window:end], rowvar=False)))[::-1]
        out[end - 1] = ev[:top].sum() / ev[:total].sum()
    return pd.Series(out, R.index)


def align(signal, target_index, lag=0):
    """
    把信号对齐到目标资产的交易日：对目标日 d，取信号在「d − lag 个自然日」当天或之前最后一个值。
        lag=0  同日对齐（原文做法）
        lag=1  只用目标日之前的信号日 —— 美国收盘晚于亚洲收盘，美国 d 日的数据
               亚洲 d 日收盘时还不知道，用于亚洲资产时应取 lag=1
    """
    s = pd.Series(signal).astype(float).sort_index()
    looked = s.reindex(pd.DatetimeIndex(target_index) - pd.Timedelta(days=lag), method="ffill")
    return pd.Series(looked.values, index=target_index).fillna(0).astype(bool)


# ══════════════════════════════════════════════════════════════════
# 各资产信号
# ══════════════════════════════════════════════════════════════════

def us_equity_signal(vol_index, real_yield, vol_window=YEAR, ry_window=2 * YEAR, q=0.99):
    """标普500 / 纳指100：恐慌指数 或 实际利率。"""
    a = exceeds(vol_index, vol_window, q) & rising(vol_index)
    b = exceeds(real_yield, ry_window, q) & rising(real_yield)
    idx = a.index.union(b.index)
    return a.reindex(idx, fill_value=False) | b.reindex(idx, fill_value=False)


def gold_signal(real_yield, window=63, q=0.99):
    return exceeds(real_yield, window, q) & rising(real_yield)


def oil_signal(ovx, window=YEAR, q=0.99):
    return exceeds(ovx, window, q) & rising(ovx)


def hk_signal(usdcnh, real_yield, fx_window=2 * YEAR, ry_window=YEAR, q=0.99):
    a = exceeds(usdcnh, fx_window, q)
    b = exceeds(real_yield, ry_window, q) & rising(real_yield)
    idx = a.index.intersection(b.index)
    return a.reindex(idx) & b.reindex(idx)


def a_share_signal(div_yield, bond_yield, index_close, pca, value_window=4 * YEAR, q=0.10,
                   short=21, long=63):
    """沪深300 / 中证500：股债性价比低 且 指数近 5 日下跌 且 PcaScore 短均 > 长均。"""
    spread = (pd.Series(div_yield) - pd.Series(bond_yield)).dropna()
    cheap_lost = below(spread, value_window, q)
    falling = pd.Series(index_close).pct_change(5) < 0
    herding = pca.rolling(short).mean() > pca.rolling(long).mean()
    idx = cheap_lost.index.intersection(falling.index).intersection(herding.index)
    return cheap_lost.reindex(idx) & falling.reindex(idx) & herding.reindex(idx)


# ══════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════

HORIZONS = (1, 2, 3, 5, 10)


def forward_returns(price, horizons=HORIZONS):
    """t 日收盘到 t+h 日收盘的收益。"""
    p = pd.Series(price).dropna()
    return pd.DataFrame({h: p.shift(-h) / p - 1 for h in horizons})


def event_study(signal, price, horizons=HORIZONS, period=None):
    """信号日与非信号日的未来 h 日平均收益（原文口径：逐日统计，重叠的信号日都算）。"""
    fwd = forward_returns(price, horizons)
    sig = align(signal, fwd.index)
    if period is not None:
        fwd, sig = fwd.loc[period], sig.loc[period]
    return pd.DataFrame({"预警时期": fwd[sig].mean(), "正常时期": fwd[~sig].mean()}).T


def episodes(signal, gap=10):
    """把信号日合并成独立事件：相邻信号日间隔不超过 gap 个交易日视为同一事件。"""
    days = np.flatnonzero(np.asarray(signal, bool))
    if len(days) == 0:
        return 0
    return int(1 + (np.diff(days) > gap).sum())


def episode_labels(signal, gap=10):
    """给每个信号日标上所属独立事件的编号（与 episodes 的归并规则相同），非信号日为 NaN。"""
    sig = pd.Series(signal).astype(bool)
    pos = np.flatnonzero(sig.values)
    lab = np.full(len(sig), np.nan)
    if len(pos):
        lab[pos] = np.concatenate([[0], np.cumsum(np.diff(pos) > gap)])
    return pd.Series(lab, sig.index)


def drop_worst_episode(signal, price, horizon=5, period=None, gap=10):
    """
    去掉对信号日平均收益「贡献最负」的那一次独立事件后，重算信号日的未来 horizon 日平均收益。
    用来检验结论是否只靠一次极端行情撑着。返回 (全部, 去掉后, 被去掉事件的起止日期)。
    """
    fwd = forward_returns(price, (horizon,))[horizon]
    sig = align(signal, fwd.index)
    if period is not None:
        fwd, sig = fwd.loc[period], sig.loc[period]
    lab = episode_labels(sig, gap)
    contrib = fwd[sig].groupby(lab[sig]).sum()
    worst = contrib.idxmin()
    keep = sig & (lab != worst)
    days = lab.index[lab == worst]
    return float(fwd[sig].mean()), float(fwd[keep].mean()), (days.min(), days.max())


def timing_returns(signal, returns, hold=5):
    """信号日之后 hold 个交易日空仓（收益记 0），其余满仓。"""
    r = pd.Series(returns).dropna()
    sig = align(signal, r.index).astype(int)
    flat = sig.shift(1, fill_value=0).rolling(hold, min_periods=1).max().astype(bool)
    return r.where(~flat, 0.0), flat


def placebo_event_study(signal, price, horizon=5, n=1000, seed=0, period=None, min_shift=YEAR):
    """
    安慰剂：把信号序列整体循环平移（保留信号的天数和扎堆结构，只打乱时点），
    得到「同样频率、随机时点」的信号日未来 horizon 日平均收益分布。
    返回 (真实值, 随机分布)。
    """
    fwd = forward_returns(price, (horizon,))[horizon]
    sig = align(signal, fwd.index)
    if period is not None:
        fwd, sig = fwd.loc[period], sig.loc[period]
    valid = fwd.notna().values
    f, s = fwd.values[valid], sig.values[valid]
    actual = f[s].mean()
    rng = np.random.default_rng(seed)
    dist = np.array([f[np.roll(s, rng.integers(min_shift, len(s) - min_shift))].mean() for _ in range(n)])
    return float(actual), dist
