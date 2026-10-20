import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from tailrisk import riskmodels as rm


def test_normal_var_es_known_values():
    r = np.random.default_rng(0).standard_normal(400_000)
    v, _ = rm.var_es_normal(r, 0.99)
    _, e = rm.var_es_normal(r, 0.975)
    assert v == pytest.approx(norm.ppf(0.99), abs=0.01)
    assert e == pytest.approx(norm.pdf(norm.ppf(0.975)) / 0.025, abs=0.01)     # ≈ 2.338


def test_t_and_gpd_match_student_t_tail():
    r = student_t.rvs(4, size=200_000, random_state=np.random.default_rng(1))
    q = 0.99
    x = student_t.ppf(q, 4)
    es_true = student_t.pdf(x, 4) / (1 - q) * (4 + x ** 2) / 3
    assert rm.var_es_t(r, q)[1] == pytest.approx(es_true, rel=0.05)
    assert rm.var_es_gpd(r, q)[1] == pytest.approx(es_true, rel=0.05)


def test_kupiec_and_christoffersen():
    rng = np.random.default_rng(2)
    good = rng.random(5000) < 0.01
    assert rm.kupiec(good, 0.01)[1] > 0.05
    assert rm.kupiec(rng.random(5000) < 0.03, 0.01)[1] < 1e-6
    clustered = np.zeros(5000, bool)
    for s in range(100, 5000, 500):
        clustered[s:s + 5] = True                                      # 突破成串出现
    assert rm.christoffersen(clustered)[1] < 1e-6


def test_z2_zero_when_correct_negative_when_es_too_low():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200_000)
    a = 0.025
    v = np.full_like(x, norm.ppf(1 - a))
    e = np.full_like(x, norm.pdf(norm.ppf(1 - a)) / a)
    assert abs(rm.acerbi_szekely_z2(x, v, e, a)) < 0.03
    assert rm.acerbi_szekely_z2(x, v, e * 0.7, a) < -0.3


def test_gjr_filter_constant_when_no_dynamics():
    p = {"mu": 0.0, "omega": 4.0, "alpha": 0.0, "gamma": 0.0, "beta": 0.0}
    sd, nxt = rm.gjr_filter(np.random.default_rng(4).normal(0, 0.02, 50), p)
    assert np.allclose(sd[1:], 0.02) and nxt == pytest.approx(0.02)   # √4 % = 2%


def test_rolling_forecasts_no_lookahead_and_shape():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=420)
    r = pd.Series(rng.standard_t(5, 420) * 0.01, idx)
    f1 = rm.rolling_forecasts(r, window=300, refit=40)
    r2 = r.copy(); r2.iloc[400:] *= 5
    f2 = rm.rolling_forecasts(r2, window=300, refit=40)
    cut = idx[399]
    pd.testing.assert_frame_equal(f1.loc[:cut].drop(columns=("收益", "")), f2.loc[:cut].drop(columns=("收益", "")))
    assert len(f1) == 120 and (f1.xs("ES", axis=1, level=1) > 0).all().all()
