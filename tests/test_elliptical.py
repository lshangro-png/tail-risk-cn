import numpy as np
import pytest

from tailrisk import elliptical as el
from tailrisk.marginals import rank_pit


@pytest.mark.parametrize("rho, nu", [(0.5, 4.0), (-0.3, 10.0)])
def test_t_density_integrates_to_one(rho, nu):
    """copula 密度 = 二元 t 密度 / 两个一元 t 密度；漏除一次归一化常数时积分约 0.38。"""
    n = 400
    g = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    U, V = np.meshgrid(g, g)
    dens = np.array([np.exp(el.t_loglik(rho, nu, np.array([a]), np.array([b])))
                     for a, b in zip(U.ravel()[::23], V.ravel()[::23])])
    assert dens.mean() == pytest.approx(1.0, abs=0.05)


def test_t_recovers_parameters():
    rng = np.random.default_rng(0)
    n, rho, nu = 3000, 0.4, 4.0
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    x = z / np.sqrt(rng.chisquare(nu, n) / nu)[:, None]
    r = el.t_vs_gaussian(rank_pit(x[:, 0]), rank_pit(x[:, 1]))
    assert r["rho_t"] == pytest.approx(rho, abs=0.05)
    assert 2.5 < r["nu"] < 6.5
    assert r["p_t"] < 1e-6


def test_t_lr_size_under_gaussian():
    """高斯数据下，5% 水平的拒绝率应接近名义水平（边界上用 ½χ²(1) 校准）。"""
    s = el.gaussian_null_t_test(window=250, n_sims=300, rho=0.2, seed=5)
    assert 0.01 <= (s.p_t < 0.05).mean() <= 0.10


def test_t_tail_dependence_formula():
    assert el.t_tail_dependence(0.0, el.NU_MAX) == 0.0
    # ρ=0.5, ν=4：λ = 2·t_5(−√(5·0.5/1.5)) ≈ 0.2532
    assert el.t_tail_dependence(0.5, 4.0) == pytest.approx(0.2532, abs=1e-3)
