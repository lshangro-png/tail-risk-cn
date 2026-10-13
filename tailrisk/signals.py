"""
趋势指标、预警状态机、预警事件切分与核对。
"""

from __future__ import annotations

import pandas as pd

from .copula import LAMBDA_ZERO_TOL


def dif_trend(px, fast=12, slow=26, kind="ma"):
    """
    DIF = 快线 - 慢线。>0 上涨趋势，<0 下跌趋势。

    中金原文写的是「标准 DIF 值（12 日移动平均 − 26 日移动平均）」，
    默认按字面用简单移动平均；kind="ema" 是 MACD 惯用的指数均线，作稳健性对照。
    """
    s = pd.Series(px).dropna()
    if kind == "ma":
        return s.rolling(fast).mean() - s.rolling(slow).mean()
    if kind == "ema":
        return s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    raise ValueError("kind 须为 'ma' 或 'ema'")


def warning_state(lambda_D, lambda_U, dif_a, dif_b, tol=LAMBDA_ZERO_TOL):
    """
    中金股债双杀预警状态机（dif_b 为债券）。

    触发：λ_D>0 且 λ_U=0 且（股或债 DIF<0）
    解除：λ_D=0 且 债券 DIF>0

    所有输入须在同一 index 上；返回 DataFrame，含两个条件与「预警中」状态。
    状态在 t 日收盘后才确定，交易上只能在 t+1 日使用。
    """
    sig = pd.DataFrame({"lambda_D": lambda_D, "lambda_U": lambda_U,
                        "DIF_a": dif_a, "DIF_b": dif_b})
    ld_pos = sig.lambda_D >= tol
    lu_zero = sig.lambda_U < tol
    sig["触发条件"] = ld_pos & lu_zero & ((sig.DIF_a < 0) | (sig.DIF_b < 0))
    sig["解除条件"] = ~ld_pos & (sig.DIF_b > 0)

    state, states = False, []
    for trig, rel in zip(sig["触发条件"], sig["解除条件"]):
        if not state and trig:
            state = True
        elif state and rel:
            state = False
        states.append(state)
    sig["预警中"] = states
    return sig


def episodes(state):
    """把布尔状态序列切成 [起, 止] 事件区间。持续交易日按 state 的 index 计数。"""
    state = pd.Series(state).fillna(False).astype(bool)
    blocks = (state != state.shift()).cumsum()
    rows = [(g.index[0], g.index[-1], len(g)) for _, g in state[state].groupby(blocks[state])]
    out = pd.DataFrame(rows, columns=["触发", "解除", "持续交易日"])
    return out.astype({"触发": "datetime64[ns]", "解除": "datetime64[ns]", "持续交易日": int})


# 中金原文披露的 5 次预警（沪深300 × 中债综合财富，2004/04–2020/12）
# 2008 年那次原文自认为错误信号。
CICC_EPISODES = pd.DataFrame({
    "触发": pd.to_datetime(["2004-10-20", "2006-06-14", "2008-09-12", "2013-06-20", "2016-12-12"]),
    "解除": pd.to_datetime(["2004-12-06", "2006-08-16", "2008-09-24", "2014-01-15", "2017-03-27"]),
    "原文判定": ["有效", "有效", "错误信号", "有效（钱荒）", "有效（钱荒）"],
})


def match_episodes(ours, reference=CICC_EPISODES, tolerance_days=20):
    """
    逐条核对：原文每次预警，复现里是否在 ±tolerance_days 个自然日内也有触发。
    另列出复现中多出来的（原文没有的）预警。
    """
    rows, used = [], set()
    for _, ref in reference.iterrows():
        gap = (ours["触发"] - ref["触发"]).dt.days
        near = gap[gap.abs() <= tolerance_days]
        if len(near):
            i = near.abs().idxmin()
            used.add(i)
            o = ours.loc[i]
            rows.append({**ref.to_dict(), "复现触发": o["触发"], "复现解除": o["解除"],
                         "触发偏差(天)": int(gap[i]), "复现持续交易日": o["持续交易日"]})
        else:
            rows.append({**ref.to_dict(), "复现触发": pd.NaT, "复现解除": pd.NaT,
                         "触发偏差(天)": None, "复现持续交易日": None})
    matched = pd.DataFrame(rows)
    extra = ours.drop(index=list(used))
    return matched, extra
