"""
基于（尾部）相关的分层聚类配置。

══ 研究对象 ══════════════════════════════════════════════════════

整理稿记载的东北证券（2025）方法：
    1. 各资产 ES（GPD 或非参数）
    2. 两两 Clayton 下尾相关 λ_L → 距离 d = −ln(λ_L)
    3. Ward 法生成树状结构
    4. 沿树自下而上，逐层用两资产公式合成尾部风险：
         TR_{1+2} = (w1·TR1 + w2·TR2) · (1 − w1·w2·(2/π)·arctan(d) · 2·min(TR1,TR2)/(TR1+TR2))
    5. 在权重上下限约束下最小化组合 TR

⚠ 未能取得东北证券原文。上式取自整理稿，无法核对是否抄写准确；
以下细节原文（整理稿）也未说明，本模块的取法：
    - 公式中的 w1、w2：取两簇在本节点内的相对权重（w_A/(w_A+w_B)）
    - 两簇之间的距离 d：两簇成员两两 d 的平均（平均连接）
    - λ_L = 0 时 d = −ln(0) = ∞：λ_L 下限截断为 1e-4（arctan 已接近 π/2，即视为完全分散）

对照方法：等权、波动率倒数、最小方差、HRP（López de Prado 2016，相关距离或尾部距离）、
直接最小化历史 ES（Rockafellar & Uryasev 2000 线性规划）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from .copula import ClaytonCopula
from .marginals import es_hist, rank_pit

LAMBDA_FLOOR = 1e-4


# ══════════════════════════════════════════════════════════════════
# 距离
# ══════════════════════════════════════════════════════════════════

def corr_distance(R):
    """López de Prado 的相关距离 √(½(1−ρ))，取值 [0, 1]。"""
    rho = R.corr().values
    return pd.DataFrame(np.sqrt(np.clip(0.5 * (1 - rho), 0, None)), R.columns, R.columns)


def lower_tail_matrix(R):
    """两两 Clayton 下尾相关 λ_L（窗口内秩变换，点估计）。对角为 1。"""
    cols = R.columns
    U = {c: rank_pit(R[c].values) for c in cols}
    lam = np.eye(len(cols))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            lam[i, j] = lam[j, i] = ClaytonCopula().fit(U[cols[i]], U[cols[j]]).lambda_raw
    return pd.DataFrame(lam, cols, cols)


def tail_distance(lam, kind="log"):
    """
    由 λ_L 得到距离。kind="log"：整理稿的 −ln(λ_L)（λ_L 截断于 1e-4）；
    kind="linear"：文献常用的 1 − λ_L（如 HRP 的尾部相关版本）。
    """
    L = np.clip(np.asarray(lam, float), LAMBDA_FLOOR, 1.0)
    D = -np.log(L) if kind == "log" else 1.0 - L
    np.fill_diagonal(D, 0.0)
    return pd.DataFrame(D, lam.index, lam.columns)


def tree(D, method="ward"):
    """由距离矩阵生成 scipy linkage 矩阵。"""
    return linkage(squareform(np.asarray(D, float), checks=False), method=method)


def cophenetic_corr(Z1, Z2):
    """两棵树的同表型相关：两两资产在两棵树上合并高度的相关系数。"""
    from scipy.cluster.hierarchy import cophenet

    return float(np.corrcoef(cophenet(Z1), cophenet(Z2))[0, 1])


# ══════════════════════════════════════════════════════════════════
# 整理稿方法：两资产合成公式与沿树递归
# ══════════════════════════════════════════════════════════════════

def two_asset_tr(w1, tr1, tr2, d):
    """整理稿记载的两资产尾部风险合成公式（w2 = 1 − w1）。"""
    w2 = 1.0 - w1
    shrink = w1 * w2 * (2 / np.pi) * np.arctan(d) * 2 * np.minimum(tr1, tr2) / (tr1 + tr2)
    return (w1 * tr1 + w2 * tr2) * (1 - shrink)


def tree_tr(w, tr, D, Z):
    """
    沿 linkage 树自下而上合成组合尾部风险。
    w、tr：各资产权重与单资产 TR（numpy 数组）；D：资产两两距离（numpy 数组）。
    返回根节点的 TR（对应归一化后的整体组合）。
    """
    n = len(w)
    members = {i: [i] for i in range(n)}
    weight = {i: w[i] for i in range(n)}
    risk = {i: tr[i] for i in range(n)}
    for k, (a, b, _, _) in enumerate(Z):
        a, b = int(a), int(b)
        wa, wb = weight[a], weight[b]
        tot = wa + wb
        d = D[np.ix_(members[a], members[b])].mean()
        node = n + k
        risk[node] = two_asset_tr(wa / tot if tot > 0 else 0.5, risk[a], risk[b], d)
        weight[node] = tot
        members[node] = members[a] + members[b]
    return float(risk[2 * n - 2])


def nesc_weights(R, q=0.95, lower=1e-4, upper=0.20, es_func=es_hist):
    """整理稿方法：Ward 树 + arctan 递归合成，在权重上下限内最小化组合 TR。"""
    lam = lower_tail_matrix(R)
    D = tail_distance(lam, "log")
    Z = tree(D, "ward")
    tr = np.array([es_func(R[c], q) for c in R.columns])
    Dv = D.values
    n = R.shape[1]
    res = minimize(lambda w: tree_tr(w, tr, Dv, Z), x0=np.full(n, 1 / n), method="SLSQP",
                   bounds=[(lower, upper)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 300, "ftol": 1e-10})
    return pd.Series(res.x / res.x.sum(), R.columns)


# ══════════════════════════════════════════════════════════════════
# 对照方法
# ══════════════════════════════════════════════════════════════════

def equal_weights(R):
    return pd.Series(1 / R.shape[1], R.columns)


def inverse_vol(R):
    iv = 1 / R.std()
    return iv / iv.sum()


def min_variance(R, lower=0.0, upper=0.20):
    S = R.cov().values
    n = S.shape[0]
    res = minimize(lambda w: w @ S @ w, x0=np.full(n, 1 / n), method="SLSQP",
                   bounds=[(lower, upper)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 300, "ftol": 1e-14})
    return pd.Series(res.x / res.x.sum(), R.columns)


def min_cvar(R, q=0.95, lower=0.0, upper=0.20):
    """
    直接最小化历史 ES（Rockafellar & Uryasev 2000）：
        min α + 1/((1−q)T)·Σ z_t   s.t.  z_t ≥ −r_t·w − α，z_t ≥ 0，Σw = 1，lower ≤ w ≤ upper
    """
    X = R.values
    T, n = X.shape
    c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1 / ((1 - q) * T))])
    A_ub = np.hstack([-X, -np.ones((T, 1)), -np.eye(T)])
    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)])[None, :]
    bounds = [(lower, upper)] * n + [(None, None)] + [(0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(T), A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    return pd.Series(res.x[:n] / res.x[:n].sum(), R.columns)


def hrp_weights(R, D=None, method="single"):
    """
    López de Prado（2016）分层风险平价：按树的叶序做准对角化，
    再递归二分，每次按两半的逆方差组合方差反比分配权重。D 缺省时用相关距离。
    """
    D = corr_distance(R) if D is None else D
    order = list(R.columns[leaves_list(tree(D, method))])
    S = R.cov()

    def cluster_var(items):
        sub = S.loc[items, items].values
        ivp = 1 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            left, right = c[: len(c) // 2], c[len(c) // 2:]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return w.reindex(R.columns)


def cap_weights(w, upper=0.20):
    """把超过上限的权重截到上限，超出部分按比例分给未触顶的资产，直到没有资产超限。"""
    w = pd.Series(w, dtype=float) / pd.Series(w, dtype=float).sum()
    for _ in range(len(w)):
        over = w > upper + 1e-12
        if not over.any():
            break
        excess = (w[over] - upper).sum()
        w[over] = upper
        free = ~over & (w < upper)
        w[free] += excess * w[free] / w[free].sum()
    return w


def block_bootstrap_sharpe_diff(r1, r2, rf=0.02, block=20, n_boot=2000, seed=0, periods=244):
    """
    两条日收益序列超额夏普之差的分块自助法置信区间（按连续日期块重抽样，保留波动率聚集等时间序列相关）。
    返回 (点估计, 2.5% 分位, 97.5% 分位, 差值 ≤ 0 的比例)。
    """
    x = pd.concat([r1, r2], axis=1).dropna().values
    T = len(x)
    rng = np.random.default_rng(seed)

    def sharpe(a):
        return (a.mean() * periods - rf) / (a.std() * np.sqrt(periods))

    point = sharpe(x[:, 0]) - sharpe(x[:, 1])
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, T - block, T // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        diffs.append(sharpe(x[idx, 0]) - sharpe(x[idx, 1]))
    diffs = np.array(diffs)
    return float(point), float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975)), float((diffs <= 0).mean())


# ══════════════════════════════════════════════════════════════════
# 滚动回测
# ══════════════════════════════════════════════════════════════════

def rolling_backtest(returns, strategies, lookback=250, rebalance="M", start=None):
    """
    每月末用过去 lookback 个交易日的收益计算各策略权重，从下一交易日起持有到下次调仓
    （期间每日再平衡到目标权重）。返回 (日收益 DataFrame, {策略: 权重 DataFrame})。
    """
    R = returns.dropna()
    ends = R.groupby(R.index.to_period(rebalance)).tail(1).index
    ends = [t for t in ends if R.index.get_loc(t) + 1 >= lookback and (start is None or t >= pd.Timestamp(start))]
    weights = {name: {} for name in strategies}
    for t in ends:
        win = R.loc[:t].tail(lookback)
        for name, f in strategies.items():
            weights[name][t] = f(win)
    out, W = {}, {}
    for name in strategies:
        wd = pd.DataFrame(weights[name]).T
        wd = wd.reindex(R.index, method="ffill").shift(1).dropna()
        W[name] = pd.DataFrame(weights[name]).T
        out[name] = (wd * R.reindex(wd.index)).sum(axis=1)
    return pd.DataFrame(out), W


def turnover(W):
    """平均单边换手率（每次调仓权重变化绝对值之和的一半，按调仓次数平均）。"""
    return float(W.diff().abs().sum(axis=1).iloc[1:].mean() / 2)


# ══════════════════════════════════════════════════════════════════
# 风险平价与尾部风险平价（风险预算）
# ══════════════════════════════════════════════════════════════════
#
# 风险预算的凸优化形式（Spinu 2013；Mausser & Romanko 2018；Roncalli 2013）：
#     y* = argmin_{y>0}  R(y) − Σ b_i ln y_i，  w = y* / Σ y*
# 对正齐次、凸的风险度量 R，最优点满足 y_i ∂R/∂y_i = b_i，即各资产的风险贡献之比等于预算之比。
# R 取波动率 → 普通风险平价（ERC）；R 取 ES → 尾部风险平价（TRP）。
# 避险资产在组合尾部日往往上涨、贡献为负，简单的迭代法会发散；凸优化形式在最优点保证各贡献为正。

def erc_weights(R, budgets=None):
    """普通风险平价：各资产对组合波动率的贡献相等（或按 budgets）。"""
    import cvxpy as cp

    S = R.cov().values
    n = S.shape[0]
    b = np.full(n, 1 / n) if budgets is None else np.asarray(budgets, float)
    scale = 1 / np.sqrt(np.diag(S).mean())                     # 数值缩放，不影响解
    y = cp.Variable(n, pos=True)
    cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(S * scale ** 2)) - b @ cp.log(y))).solve(solver="CLARABEL")
    w = np.maximum(y.value, 0)
    return pd.Series(w / w.sum(), R.columns)


def es_parity_weights(scenarios, q=0.95, budgets=None):
    """
    尾部风险平价：各资产对组合 ES 的贡献相等（或按 budgets）。
    scenarios：T×n 收益情景（历史收益，或 GARCH 过滤后的情景）。ES 用 Rockafellar–Uryasev 表达式。
    """
    import cvxpy as cp

    X = np.asarray(scenarios, float)
    X = X / X.std(axis=0).mean()                                  # 数值缩放，不影响解
    T, n = X.shape
    b = np.full(n, 1 / n) if budgets is None else np.asarray(budgets, float)
    y = cp.Variable(n, pos=True)
    alpha = cp.Variable()
    u = cp.Variable(T, nonneg=True)                               # u_t ≥ 组合损失超过 α 的部分
    prob = cp.Problem(cp.Minimize(alpha + cp.sum(u) / ((1 - q) * T) - b @ cp.log(y)), [u >= -X @ y - alpha])
    try:
        prob.solve(solver="CLARABEL")                             # 情景数在一千左右时快而准
    except cp.error.SolverError:
        prob.solve(solver="SCS", eps=1e-8, max_iters=200_000)     # 情景更多时 Clarabel 会失败
    w = np.maximum(y.value, 0)
    cols = scenarios.columns if hasattr(scenarios, "columns") else range(n)
    return pd.Series(w / w.sum(), cols)


def es_contributions(w, scenarios, q=0.95):
    """
    组合 ES 的欧拉分解：RC_i = −w_i · E[r_i | 组合落在最差的 (1−q) 尾部]，各项之和等于组合 ES。
    """
    X = np.asarray(scenarios, float)
    w = np.asarray(w, float)
    port = X @ w
    k = max(int(np.ceil((1 - q) * len(port))), 1)
    tail = np.argsort(port)[:k]
    return -w * X[tail].mean(axis=0)


def vol_contributions(w, R):
    S = np.asarray(R.cov(), float)
    w = np.asarray(w, float)
    return w * (S @ w) / np.sqrt(w @ S @ w)


def fhs_scenarios(R):
    """
    GARCH 过滤的历史情景：各资产分别拟合 GJR-GARCH(1,1)，得到标准化残差 z_t，
    再乘以各自明日的预测波动率。保留历史上各资产残差的同日联动（依赖结构），
    但把尺度换成当前的波动率水平（Barone-Adesi 等 1999 的多元 FHS）。
    """
    from .riskmodels import fit_gjr, gjr_filter

    out = {}
    for c in R.columns:
        x = R[c].values
        p = fit_gjr(x)
        sd, sd_next = gjr_filter(x, p)
        out[c] = (x - p["mu"] / 100) / sd * sd_next + p["mu"] / 100
    return pd.DataFrame(out, index=R.index)
