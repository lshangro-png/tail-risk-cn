import numpy as np
import pandas as pd
import pytest

from tailrisk import crowding as cw


def test_rolling_sharpe_includes_current_month():
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    r = pd.Series(0.01, idx); r.iloc[-1] = -0.20
    s = cw.rolling_sharpe(r + np.random.default_rng(0).normal(0, 1e-3, 40), 36)
    assert s.iloc[-1] < s.iloc[-2]                          # 当月大跌立刻拉低当月的滚动夏普


def test_fit_decay_recovers_hyperbola():
    t = np.linspace(0, 30, 300)
    y = 0.8 / (1 + 0.2 * t) + np.random.default_rng(1).normal(0, 0.01, 300)
    p, r2 = cw.fit_decay(t, y, "双曲线")
    assert p[0] == pytest.approx(0.8, rel=0.05) and p[1] == pytest.approx(0.2, rel=0.1) and r2 > 0.95


def test_residual_uses_only_past_fits():
    rng = np.random.default_rng(2)
    idx = pd.date_range("1960-01-31", periods=300, freq="ME")
    s = pd.Series(rng.normal(0.5, 0.2, 300), idx)
    a = cw.crowding_residual(s, min_obs=120)
    s2 = s.copy(); s2.iloc[250:] += 5
    b = cw.crowding_residual(s2, min_obs=120)
    pd.testing.assert_series_equal(a.iloc[:130], b.iloc[:130])   # 第 250 个月之后的变化不影响此前的残差


def test_crash_table_counts():
    idx = pd.date_range("1980-01-31", periods=48, freq="ME")
    year = [0.01, 0.02, 0.0, 0.015, 0.005, 0.01, 0.03, -0.01, 0.0, 0.02, 0.01, -0.08]   # 每年只有 12 月大跌
    r = pd.Series(np.tile(year, 4), idx)
    sig = pd.Series(np.where(r < -0.05, -1.0, 1.0), idx)       # 每次大跌当月信号都低
    t = cw.crash_table(r, sig, 0, train=("1980", "1981"), test=("1982", "1983"))
    assert t["P(崩盘|拥挤)"] == 1.0 and t["P(崩盘|不拥挤)"] == 0.0
    t1 = cw.crash_table(r, sig, 1, train=("1980", "1981"), test=("1982", "1983"))
    assert t1["P(崩盘|拥挤)"] == 0.0                            # 下月口径：大跌之后的 1 月不是崩盘
