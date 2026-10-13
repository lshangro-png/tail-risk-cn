"""
边缘分布与概率积分变换（PIT）。

中金原文没有说明边缘分布怎么处理，只说尾部相关系数「使用过去 60 个交易日
数据计算」。本项目据此把**窗口内秩变换**（rank_pit，在 copula.rolling_tail_dependence
里逐窗口调用）作为主口径：只用窗口内数据，天然没有前视偏差。

下面三个全样本函数作为对照口径保留：
    pit_empirical   全样本秩变换 —— 用到了未来数据，有前视偏差
    pit_garch       全样本 GJR-GARCH 滤波后秩变换 —— 同上，且会剥离波动率联动
    pit_vol_scaled  滚动波动率标准化后全样本秩变换
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import rankdata


def rank_pit(x):
    """
    秩变换到开区间 (0,1)：rank/(n+1)，ties 取平均秩。
    除以 n+1 而非 n，避免 Copula 对数似然出现 log(0)。
    """
    return rankdata(x, method="average") / (len(x) + 1)


def pit_empirical(r):
    """全样本经验 CDF 变换。⚠ 每个点的分位都依赖全样本，含未来信息。"""
    s = pd.Series(r).dropna()
    return pd.Series(rank_pit(s.values), index=s.index)


def pit_garch(r, dist="skewt"):
    """
    GJR-GARCH 滤波 + 标准化残差的经验 PIT。⚠ 全样本拟合，含未来信息。

    危机期「股债同步暴跌」在数据上很大程度表现为两者波动率同时放大，
    GARCH 滤波恰好剥离这一层，可能把要测的东西一起剥掉。只作对照。
    """
    from arch import arch_model

    s = pd.Series(r).dropna()
    am = arch_model(s * 100, vol="GARCH", p=1, o=1, q=1, dist=dist)
    res = am.fit(disp="off", show_warning=False)
    z = res.std_resid.dropna()
    return pd.Series(rank_pit(z.values), index=z.index)


def pit_vol_scaled(r, window=60):
    """
    滚动波动率标准化后再做经验 PIT。介于上面两者之间的中间档。

    波动率用 shift(1)：当日收益不参与自己的标准化。
    但最后一步秩变换仍是全样本的。
    """
    s = pd.Series(r).dropna()
    vol = s.rolling(window, min_periods=window // 2).std().shift(1)
    z = (s / vol).dropna()
    return pd.Series(rank_pit(z.values), index=z.index)
