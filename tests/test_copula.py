"""
Copula 数学的回归测试。这些都是踩过坑的地方，改 copula.py 前后都要跑：

    pytest -q
"""

import numpy as np
import pytest

from tailrisk import copula as cp
from tailrisk.marginals import rank_pit


def _grid(n=800):
    g = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    return np.meshgrid(g, g)


@pytest.mark.parametrize("C, theta", [
    (cp.ClaytonCopula, 0.01), (cp.ClaytonCopula, 2.0),
    (cp.GumbelCopula, 1.0), (cp.GumbelCopula, 1.8),
])
def test_density_integrates_to_one(C, theta):
    U, V = _grid()
    assert np.exp(C.loglik_terms(theta, U, V)).mean() == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize("C, theta0", [(cp.ClaytonCopula, 1e-9), (cp.GumbelCopula, 1.0)])
def test_independence_loglik_is_zero(C, theta0):
    U, V = _grid(50)
    assert np.abs(C.loglik_terms(theta0, U, V)).max() < 1e-6


def test_clayton_recovers_theta():
    """条件分布法抽 Clayton(θ=2) 样本，MLE 应估回 θ≈2，λ_D≈2^-0.5。"""
    rng = np.random.default_rng(0)
    n, th = 4000, 2.0
    u, w = rng.uniform(size=n), rng.uniform(size=n)
    v = (u ** (-th) * (w ** (-th / (1 + th)) - 1) + 1) ** (-1 / th)
    c = cp.ClaytonCopula().fit(u, v)
    assert c.theta == pytest.approx(th, rel=0.1)
    assert c.lambda_raw == pytest.approx(2 ** -0.5, abs=0.03)


def test_no_overflow_at_extreme_ranks():
    """θ 大、u 极小时 u^-θ 会溢出；log 空间实现必须给出有限似然。"""
    u = np.array([1e-6, 0.5, 1 - 1e-6])
    assert np.isfinite(cp.ClaytonCopula.loglik_terms(30.0, u, u)).all()
    assert np.isfinite(cp.GumbelCopula.loglik_terms(30.0, u, u)).all()


def test_negative_dependence_gives_zero_lambda():
    """负相关窗口：两族都应落在独立边界上，λ=0（两种判零口径都是）。"""
    rng = np.random.default_rng(1)
    z1 = rng.standard_normal(60)
    z2 = -0.6 * z1 + 0.8 * rng.standard_normal(60)
    res = cp.fit_window(z1, z2)
    for rule in cp.ZERO_RULES:
        assert cp.zero_out(res["lambda_D_raw"], res["p_D"], rule) == 0
        assert cp.zero_out(res["lambda_U_raw"], res["p_U"], rule) == 0


def test_lrtest_false_positive_rate_under_independence():
    """LR 口径下，独立数据的 λ_D>0 占比不应超过名义 5%（χ²(1) 偏保守）。"""
    b = cp.noise_baseline(window=60, n_rep=600, rho=0.0, seed=7, zero_rule="lrtest")
    assert b["lambda_D>0占比"] <= 0.07


def test_positive_correlation_inflates_lambda_without_tail_dependence():
    """
    已知局限的「回归测试」：高斯 copula（无尾部相关）ρ=0.5 时，
    λ_D>0 仍几乎总是成立。若这条失败，说明判零逻辑变了，结论需要重审。
    """
    b = cp.noise_baseline(window=60, n_rep=300, rho=0.5, seed=8, zero_rule="lrtest")
    assert b["lambda_D>0占比"] > 0.85


def test_rank_pit_open_interval():
    x = np.array([3.0, 1.0, 2.0, 2.0])
    u = rank_pit(x)
    assert (u > 0).all() and (u < 1).all()
    assert u[2] == u[3]                      # ties 取平均秩


def test_rolling_window_has_no_lookahead():
    """改动窗口之后的数据，不应影响此前任何一天的估计。"""
    import pandas as pd

    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=150)
    x, y = pd.Series(rng.standard_normal(150), idx), pd.Series(rng.standard_normal(150), idx)
    r1 = cp.rolling_tail_dependence(x, y, window=60)
    y2 = y.copy()
    y2.iloc[120:] *= -5
    r2 = cp.rolling_tail_dependence(x, y2, window=60)
    cut = idx[119]
    pd.testing.assert_frame_equal(r1.loc[:cut], r2.loc[:cut])
