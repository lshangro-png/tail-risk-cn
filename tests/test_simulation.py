import numpy as np
import pytest
from scipy.stats import kurtosis, ks_2samp

from tailrisk import simulation as sm


def _skewed_hist(T=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0, 0], [[1, .6, .3], [.6, 1, .4], [.3, .4, 1]], T)
    H = np.exp(0.6 * z)
    return (H - H.mean(0)) / H.std(0) * np.array([.01, .02, .015])


def test_gtja_recursion_inflates_later_variances():
    """整理稿记载的递推式：第 1 位资产不受影响，后面与之相关的资产方差被高估。"""
    H = _skewed_hist()
    X = sm.gtja_recursion(H, 100_000, np.random.default_rng(1))
    ratio = np.diag(np.cov(X, rowvar=False)) / np.diag(np.cov(H, rowvar=False))
    assert ratio[0] < 1.10
    assert ratio[1] > 1.20 and ratio[2] > 1.10


@pytest.mark.parametrize("method", [sm.gaussian_copula_kde, sm.iman_conover])
def test_correct_methods_preserve_marginals_and_order_free(method):
    H = _skewed_hist()
    rng = np.random.default_rng(2)
    X = method(H, 100_000, rng)
    ratio = np.diag(np.cov(X, rowvar=False)) / np.diag(np.cov(H, rowvar=False))
    assert np.all(np.abs(ratio - 1) < 0.08)                       # 仅 KDE 带宽带来的少量平滑
    for j in range(3):
        assert ks_2samp(X[:, j], H[:, j]).statistic < 0.03
        # 样本峰度在厚尾下抽样波动很大，KDE 平滑也会略降峰度，容差取 25%；
        # 作为对照，递推式会把第 2 位资产的峰度稀释约 40%
        assert kurtosis(X[:, j]) == pytest.approx(kurtosis(H[:, j]), rel=0.25)
    # 顺序无关：倒序排列后，各资产的方差误差不变
    Xr = method(H[:, ::-1], 100_000, rng)[:, ::-1]
    ratio_r = np.diag(np.cov(Xr, rowvar=False)) / np.diag(np.cov(H, rowvar=False))
    np.testing.assert_allclose(ratio, ratio_r, atol=0.05)


def test_kde_ppf_inverts_cdf():
    x = np.random.default_rng(3).standard_normal(500)
    q = sm.kde_ppf(x, np.array([0.05, 0.5, 0.95]))
    assert q[0] < q[1] < q[2]
    assert q[1] == pytest.approx(np.median(x), abs=0.1)


def test_simulate_horizon_shape():
    H = _skewed_hist(300)
    P = sm.simulate_horizon(sm.iman_conover, H, horizon=20, n_paths=50, rng=np.random.default_rng(4))
    assert P.shape == (50, 20, 3)


def test_simplex_grid():
    g = sm.simplex_grid(3, 0.25)
    assert len(g) == 15                                      # C(4+2, 2)
    np.testing.assert_allclose(g.sum(axis=1), 1.0)
    assert (g >= 0).all()


def test_path_risk_decomposition_and_values():
    rng = np.random.default_rng(5)
    paths = rng.normal(0.0005, 0.01, (2000, 20, 2))
    W = sm.simplex_grid(2, 0.1)
    r = sm.path_risk(paths, W)
    np.testing.assert_allclose(r["TR"], r["IH"] + r["EH"])     # 总风险 = 期内 + 期末（互斥）
    # 对单一资产直接按对数路径计算，应与向量化结果一致
    logP = np.log(np.cumprod(1 + paths[:, :, 0], axis=1))
    ih = np.minimum(logP.min(1), 0) <= -0.01
    eh = ~ih & (logP[:, -1] <= 0.01)
    i = int(np.flatnonzero((W[:, 0] == 1.0))[0])
    assert r["IH"][i] == pytest.approx(ih.mean()) and r["EH"][i] == pytest.approx(eh.mean())


def test_optimizer_picks_obvious_asset():
    """一个资产稳定上涨（几乎不会触发风险），另一个高波动：应全部配给前者。"""
    rng = np.random.default_rng(6)
    steady = np.full((1500, 20, 1), 0.001) + rng.normal(0, 0.0002, (1500, 20, 1))
    wild = rng.normal(0, 0.03, (1500, 20, 1))
    w, risk = sm.gtja_optimal_weights(np.concatenate([steady, wild], axis=2), sm.simplex_grid(2, 0.05))
    assert w[0] == pytest.approx(1.0) and risk["TR"] < 0.05
