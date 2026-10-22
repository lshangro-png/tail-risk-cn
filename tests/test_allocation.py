import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from tailrisk import allocation as al
from tailrisk.marginals import es_gpd, es_hist


def _returns(n=600, k=4, seed=0):
    rng = np.random.default_rng(seed)
    C = 0.5 * np.ones((k, k)) + 0.5 * np.eye(k)
    X = rng.multivariate_normal(np.zeros(k), C, n) * np.linspace(0.005, 0.02, k)
    return pd.DataFrame(X, columns=[f"a{i}" for i in range(k)])


def test_two_asset_formula_limits():
    # d=0（完全尾部相依）：无分散效果，等于线性加权
    assert al.two_asset_tr(0.5, 0.02, 0.04, 0.0) == pytest.approx(0.03)
    # 单一资产：w1=1 时就是该资产自身的 TR
    assert al.two_asset_tr(1.0, 0.02, 0.04, 5.0) == pytest.approx(0.02)
    # 距离越大，合成风险越小
    assert al.two_asset_tr(0.5, 0.03, 0.03, 9.0) < al.two_asset_tr(0.5, 0.03, 0.03, 0.5)


def test_tree_tr_reduces_to_formula_for_two_assets():
    Z = np.array([[0, 1, 1.0, 2]])
    D = np.array([[0.0, 2.0], [2.0, 0.0]])
    tr = np.array([0.02, 0.05])
    assert al.tree_tr(np.array([0.3, 0.7]), tr, D, Z) == pytest.approx(al.two_asset_tr(0.3, 0.02, 0.05, 2.0))


def test_min_cvar_beats_alternatives():
    R = _returns()
    w = al.min_cvar(R, q=0.95, upper=1.0)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()
    es_opt = es_hist(R @ w)
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = rng.dirichlet(np.ones(R.shape[1]))
        assert es_hist(R @ v) >= es_opt - 1e-4


def test_weight_functions_respect_bounds():
    R = _returns(k=6)
    for f in (al.min_variance, al.min_cvar):
        w = f(R, upper=0.25)
        assert w.sum() == pytest.approx(1.0) and w.max() <= 0.25 + 1e-6
    h = al.hrp_weights(R)
    assert h.sum() == pytest.approx(1.0) and (h > 0).all()


def test_es_gpd_matches_student_t():
    """t(4) 分布 99% ES 的理论值；GPD 外推应与之接近（大样本）。"""
    rng = np.random.default_rng(3)
    r = student_t.rvs(4, size=200_000, random_state=rng)
    q, nu = 0.99, 4
    x = student_t.ppf(q, nu)
    es_true = student_t.pdf(x, nu) / (1 - q) * (nu + x ** 2) / (nu - 1)
    assert es_gpd(r, q=q, threshold=0.95) == pytest.approx(es_true, rel=0.05)
    assert es_hist(r, q=q) == pytest.approx(es_true, rel=0.05)


def test_cap_weights():
    w = al.cap_weights(pd.Series([0.5, 0.3, 0.1, 0.1]), upper=0.3)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.3 + 1e-9
    assert w.iloc[2] == pytest.approx(w.iloc[3])            # 超额按原比例分配


def test_erc_equalizes_vol_contributions():
    R = _returns(k=5)
    w = al.erc_weights(R)
    rc = al.vol_contributions(w, R)
    np.testing.assert_allclose(rc / rc.sum(), 0.2, atol=1e-3)


def test_es_parity_equalizes_es_contributions_with_hedge_asset():
    """含一个与股票负相关的「避险」资产：其 ES 贡献在等权时为负，TRP 解仍应使各贡献相等且为正。"""
    rng = np.random.default_rng(7)
    T = 3000
    eq = rng.standard_t(4, (T, 2)) * 0.012 + rng.standard_t(4, (T, 1)) * 0.008
    hedge = -0.3 * eq[:, :1] + rng.normal(0, 0.003, (T, 1))
    X = pd.DataFrame(np.hstack([eq, hedge]), columns=["股1", "股2", "避险"])
    assert al.es_contributions(np.full(3, 1 / 3), X)[2] < 0
    w = al.es_parity_weights(X, q=0.95)
    rc = al.es_contributions(w.values, X, q=0.95)
    assert (rc > 0).all()
    np.testing.assert_allclose(rc / rc.sum(), 1 / 3, atol=0.03)


def test_es_parity_close_to_erc_for_gaussian():
    """椭圆分布下 ES 贡献与波动率贡献成比例，两种风险平价的权重应几乎相同。"""
    R = _returns(n=20000, k=4, seed=8)
    np.testing.assert_allclose(al.es_parity_weights(R).values, al.erc_weights(R).values, atol=0.02)
