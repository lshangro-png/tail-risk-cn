"""
日频回测：风险平价基准、预警叠加、绩效统计。

所有回测都在**日频**上做。信号若是隔 step 天重估的，先 reindex 到日频再前向填充，
不能直接在稀疏的重估点上累乘收益（那样只复利了 1/step 的交易日）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 244


def to_daily(signal, daily_index):
    """把稀疏重估点上的信号扩展到日频：每天取最近一次重估的结果。"""
    return pd.Series(signal).reindex(daily_index, method="ffill")


def risk_parity_weights(returns, lookback=60, target_vol=0.08, max_leverage=None,
                        rebalance="ME"):
    """
    月末调仓的风险平价权重（中金原文：月末调仓、目标波动 8%）。

    两资产时等风险贡献的解恰好是波动率倒数加权（与相关系数无关）；
    多资产时这里同样用波动率倒数（朴素风险平价）。
    再整体缩放到目标波动，max_leverage 限制总杠杆。

    权重在调仓日收盘计算，从下一交易日起生效（shift(1)）。
    原文没说明协方差回看期与杠杆上限，这两个参数需做敏感性检验。
    """
    rets = returns.dropna()
    vol = rets.rolling(lookback).std()
    rebal_dates = rets.groupby(rets.index.to_period(rebalance[0])).tail(1).index

    w = {}
    for t in rebal_dates:
        if vol.loc[t].isna().any():
            continue
        raw = 1 / vol.loc[t]
        raw = raw / raw.sum()
        cov = rets.loc[:t].tail(lookback).cov() * TRADING_DAYS
        port_vol = float(np.sqrt(raw @ cov @ raw))
        k = target_vol / port_vol if target_vol else 1.0
        if max_leverage is not None:
            k = min(k, max_leverage)
        w[t] = raw * k
    w = pd.DataFrame(w).T
    return w.reindex(rets.index, method="ffill").shift(1).dropna()


def portfolio_returns(weights, returns, cash_rate=0.02):
    """
    组合日收益。权重和不为 1 的部分按现金利率计息（>1 即借入资金，按同一利率付息）。
    月内权重视为每日再平衡到目标权重（简化，忽略漂移）。
    """
    w = weights.reindex(returns.index).dropna()
    r = returns.reindex(w.index)
    cash_w = 1 - w.sum(axis=1)
    return (w * r).sum(axis=1) + cash_w * cash_rate / TRADING_DAYS


def overlay_warning(base_returns, warning, cash_rate=0.02):
    """
    预警叠加：t 日收盘处于预警 → t+1 日全部转为货币资产。
    warning 为日频布尔序列（t 日收盘时的状态）。
    """
    in_warn = (warning.reindex(base_returns.index, fill_value=False).astype(bool)
               .shift(1, fill_value=False))
    return pd.Series(np.where(in_warn, cash_rate / TRADING_DAYS, base_returns),
                     index=base_returns.index)


def max_drawdown(r):
    nav = (1 + r.dropna()).cumprod()
    return float((nav / nav.cummax() - 1).min())


def perf_stats(r, rf=0.02):
    """
    绩效统计。「收益波动比」= 年化收益/年化波动，与中金原文同口径；
    但空仓期按现金利率计息、波动为 0，会抬高这个比值，
    所以另给扣除无风险利率 rf 后的「超额夏普」，比较策略时以它为准。
    """
    r = r.dropna()
    yrs = len(r) / TRADING_DAYS
    ann = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    mdd = max_drawdown(r)
    return {"年化收益": ann, "年化波动": vol, "收益波动比": ann / vol if vol else np.nan,
            "超额夏普": (ann - rf) / vol if vol else np.nan,
            "最大回撤": mdd, "卡玛比率": ann / abs(mdd) if mdd else np.nan}


def yearly_table(returns_df):
    """逐年收益与年内最大回撤（与中金原文的分年表同口径）。"""
    out = {}
    for name, r in returns_df.items():
        g = r.dropna().groupby(r.dropna().index.year)
        out[(name, "年收益")] = g.apply(lambda x: (1 + x).prod() - 1)
        out[(name, "年最大回撤")] = g.apply(max_drawdown)
    return pd.DataFrame(out)


def pairwise_cash_overlay(returns, base_weights, warnings, cash_rate=0.02):
    """
    中金多资产回撤控制：每日统计两两预警的触发数 n（t 日收盘状态，t+1 日执行）。
        n=0   按基准权重配置
        n=1   触发预警的两个资产仓位转为货币，其余资产保持基准权重
        n≥2   100% 货币
    base_weights：{资产: 权重}；warnings：{(资产a, 资产b): 日频布尔序列}。
    基准权重视为每日再平衡（与 portfolio_returns 一致）。
    """
    idx = returns.index
    w0 = pd.Series(base_weights, dtype=float)
    states = {pair: s.reindex(idx, fill_value=False).astype(bool).shift(1, fill_value=False)
              for pair, s in warnings.items()}
    n = sum(s.astype(int) for s in states.values())
    weights = pd.DataFrame(np.tile(w0.values, (len(idx), 1)), index=idx, columns=w0.index)
    for pair, s in states.items():
        single = s & (n == 1)
        weights.loc[single, list(pair)] = 0.0
    weights.loc[n >= 2, :] = 0.0
    r = (weights * returns[w0.index]).sum(axis=1) + (1 - weights.sum(axis=1)) * cash_rate / TRADING_DAYS
    return r, n
