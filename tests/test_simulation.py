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
