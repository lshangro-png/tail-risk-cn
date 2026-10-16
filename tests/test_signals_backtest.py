import numpy as np
import pandas as pd

from tailrisk import backtest as bt
from tailrisk import data
from tailrisk import signals as sg


def test_to_returns_does_not_fill_gaps():
    px = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 1.1, 1.2, 1.3]})
    r = data.to_returns(px)["a"]
    assert np.isnan(r.loc[2]) and np.isnan(r.loc[3])     # 缺失日及其后一日都不应有收益


def test_state_machine_trigger_and_release():
    idx = pd.bdate_range("2021-01-01", periods=6)
    ld = pd.Series([0, .2, .2, 0, 0, 0], idx, dtype=float)
    lu = pd.Series([0, 0, .1, 0, 0, 0], idx, dtype=float)
    dif_a = pd.Series([1, -1, -1, -1, -1, -1], idx, dtype=float)
    dif_b = pd.Series([1, 1, 1, -1, 1, 1], idx, dtype=float)
    s = sg.warning_state(ld, lu, dif_a, dif_b)
    # 第 2 天触发；第 3 天 λ_U>0 但已在预警中（只看解除条件）；
    # 第 4 天 λ_D=0 但债 DIF<0 不解除；第 5 天解除
    assert s["预警中"].tolist() == [False, True, True, True, False, False]


def test_episodes_and_matching():
    idx = pd.bdate_range("2013-06-17", periods=10)
    state = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], idx).astype(bool)
    ep = sg.episodes(state)
    assert len(ep) == 1 and ep.loc[0, "持续交易日"] == 3
    m, extra = sg.match_episodes(ep)
    assert m.loc[m["触发"] == "2013-06-20", "触发偏差(天)"].item() == 0
    assert extra.empty
    assert sg.episodes(pd.Series(False, idx)).empty


def test_overlay_uses_signal_next_day():
    idx = pd.bdate_range("2021-01-01", periods=4)
    base = pd.Series([0.01, 0.01, 0.01, 0.01], idx)
    warn = pd.Series([False, True, False, False], idx)
    out = bt.overlay_warning(base, warn, cash_rate=0.0)
    assert out.tolist() == [0.01, 0.01, 0.0, 0.01]        # t 日预警 → t+1 日空仓


def test_to_daily_forward_fills_sparse_signal():
    daily = pd.bdate_range("2021-01-01", periods=10)
    sparse = pd.Series([True, False], index=daily[[0, 5]])
    d = bt.to_daily(sparse, daily)
    assert d.iloc[:5].all() and not d.iloc[5:].any()


def test_risk_parity_weights_are_lagged_and_inverse_vol():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.DataFrame({"e": rng.normal(0, .02, 300), "b": rng.normal(0, .002, 300)}, idx)
    w = bt.risk_parity_weights(r, lookback=60, target_vol=None)
    month_end = r.groupby(r.index.to_period("M")).tail(1).index[3]
    nxt = r.index[r.index.get_loc(month_end) + 1]
    vol = r.loc[:month_end].tail(60).std()
    expect = (1 / vol) / (1 / vol).sum()
    np.testing.assert_allclose(w.loc[nxt].values, expect.values, rtol=1e-6)
    assert w.loc[month_end].ne(w.loc[nxt]).any()            # 调仓日当天仍用旧权重


def test_pairwise_cash_overlay_rules():
    idx = pd.bdate_range("2021-01-01", periods=4)
    r = pd.DataFrame({"股": 0.01, "债": 0.002, "金": 0.005}, index=idx)
    w = {"股": 0.2, "债": 0.6, "金": 0.2}
    warn = {("股", "债"): pd.Series([False, True, True, False], idx),
            ("股", "金"): pd.Series([False, False, True, False], idx)}
    out, n = bt.pairwise_cash_overlay(r, w, warn, cash_rate=0.0)
    # t 日状态 → t+1 日执行：第 2 天 n=0；第 3 天 n=1（股债转现金，只剩金 0.2）；第 4 天 n=2 全现金
    assert n.tolist() == [0, 0, 1, 2]
    np.testing.assert_allclose(out.values, [0.0042, 0.0042, 0.2 * 0.005, 0.0])
