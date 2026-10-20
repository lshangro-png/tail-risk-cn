import numpy as np
import pandas as pd
import pytest

from tailrisk import warning as wn


def test_exceeds_uses_only_past_data():
    idx = pd.bdate_range("2020-01-01", periods=300)
    x = pd.Series(np.arange(300, dtype=float), idx)          # 单调上升：每天都创新高
    e = wn.exceeds(x, 100, 0.99)
    assert not e.iloc[:100].any()                           # 窗口未满不出信号
    assert e.iloc[100:].all()
    x2 = x.copy(); x2.iloc[200] = -1                         # 当日骤降不应触发
    assert not wn.exceeds(x2, 100, 0.99).iloc[200]


def test_rising_slope():
    x = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
    r = wn.rising(x, 5)
    assert r.iloc[4] and not r.iloc[9]


def test_pca_score_bounds_and_herding():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=200)
    common = rng.standard_normal((200, 1))
    indep = pd.DataFrame(rng.standard_normal((200, 16)), idx)
    herd = pd.DataFrame(common * 3 + rng.standard_normal((200, 16)) * 0.3, idx)
    a, b = wn.pca_score(indep).dropna(), wn.pca_score(herd).dropna()
    assert ((a > 0) & (a <= 1)).all() and ((b > 0) & (b <= 1)).all()
    assert b.mean() > 0.8 > a.mean()                        # 同向走势时前两大主成分占比高


def test_timing_flat_after_signal():
    idx = pd.bdate_range("2020-01-01", periods=12)
    r = pd.Series(0.01, idx)
    sig = pd.Series(False, idx); sig.iloc[2] = True
    out, flat = wn.timing_returns(sig, r, hold=5)
    assert flat.tolist() == [False] * 3 + [True] * 5 + [False] * 4   # 信号次日起空仓 5 天
    assert (out[flat] == 0).all()


def test_align_lag_and_ffill():
    us = pd.Series([False, True, False], pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    asia = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    assert wn.align(us, asia).tolist() == [True, False, False]
    assert wn.align(us, asia, lag=1).tolist() == [False, True, False]   # 美国 1/3 的信息亚洲 1/6 才能用


def test_episodes():
    s = np.zeros(100, bool); s[[5, 6, 8, 40, 41, 90]] = True
    assert wn.episodes(s, gap=10) == 3


def test_episode_labels_and_drop_worst():
    idx = pd.bdate_range("2020-01-01", periods=100)
    sig = pd.Series(False, idx); sig.iloc[[5, 6, 40, 41, 80]] = True
    lab = wn.episode_labels(sig)
    assert lab.dropna().tolist() == [0, 0, 1, 1, 2]
    price = pd.Series(100.0, idx); price.iloc[46:] = 50.0        # 第二次事件之后暴跌
    allm, kept, (a, b) = wn.drop_worst_episode(sig, price, horizon=5)
    assert allm < 0 and kept == pytest.approx(0.0) and a == idx[40] and b == idx[41]
