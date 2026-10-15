import numpy as np
import pytest

from tailrisk import sjc
from tailrisk.marginals import rank_pit


def _jc_cdf(u, v, tu, tl):
    k, g = 1 / np.log2(2 - tu), -1 / np.log2(tl)
    A = (1 - (1 - u) ** k) ** (-g) + (1 - (1 - v) ** k) ** (-g) - 1
    return 1 - (1 - A ** (-1 / g)) ** (1 / k)


@pytest.mark.parametrize("tu, tl", [(0.3, 0.6), (0.05, 0.2), (0.7, 0.1)])
def test_jc_density_matches_cdf_mixed_derivative(tu, tl):
    e = 1e-4
    for u, v in [(0.2, 0.3), (0.9, 0.85), (0.05, 0.6)]:
        fd = (_jc_cdf(u + e, v + e, tu, tl) - _jc_cdf(u + e, v - e, tu, tl)
              - _jc_cdf(u - e, v + e, tu, tl) + _jc_cdf(u - e, v - e, tu, tl)) / (4 * e * e)
        assert np.exp(sjc.jc_logpdf(np.array(u), np.array(v), tu, tl)) == pytest.approx(fd, rel=1e-3)


@pytest.mark.parametrize("tu, tl", [(0.3, 0.6), (0.7, 0.1), (1e-4, 1e-4)])
def test_sjc_density_integrates_to_one(tu, tl):
    n = 500
    g = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    U, V = np.meshgrid(g, g)
    assert np.exp(sjc.sjc_logpdf(U, V, tu, tl)).mean() == pytest.approx(1.0, abs=0.01)


def test_finite_in_upper_corner_with_large_tau():
    """κ 大时 (1−u)^κ 会下溢，曾导致 w=0、似然为 +inf，优化器被带偏。"""
    u = np.array([0.999, 0.9999, 0.5])
    assert np.isfinite(sjc.sjc_logpdf(u, u, sjc.TAU_MAX, sjc.TAU_MAX)).all()


def test_recovers_pure_lower_tail():
    """Clayton(θ=1)：λ_L = 0.5，λ_U = 0；不对称检验应强烈拒绝。"""
    rng = np.random.default_rng(0)
    n = 3000
    u, w = rng.uniform(size=n), rng.uniform(size=n)
    v = (u ** -1.0 * (w ** -0.5 - 1) + 1) ** -1.0
    r = sjc.asymmetry_test(rank_pit(u), rank_pit(v))
    assert r["tau_L"] == pytest.approx(0.5, abs=0.08)
    assert r["tau_U"] < 0.05
    assert r["p_asym"] < 1e-10


def test_asymmetry_size_under_gaussian():
    """对称的高斯数据（ρ=0.3）下，5% 水平误报率应接近名义水平。"""
    rng = np.random.default_rng(1)
    p = []
    for _ in range(200):
        z1 = rng.standard_normal(250)
        z2 = 0.3 * z1 + np.sqrt(1 - 0.09) * rng.standard_normal(250)
        p.append(sjc.asymmetry_test(rank_pit(z1), rank_pit(z2))["p_asym"])
    assert 0.01 <= np.nanmean(np.array(p) < 0.05) <= 0.10
