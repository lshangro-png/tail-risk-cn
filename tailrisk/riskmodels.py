"""
单资产 VaR / ES 的滚动预测与回测。

══ 六种方法 ══════════════════════════════════════════════════════

    历史模拟（HS）     窗口内损失的经验分位数与超出部分均值
    正态              窗口内均值、标准差
    t 分布            窗口内极大似然拟合的 Student-t
    GPD（POT）        对窗口内超过 90% 分位的损失拟合广义帕累托分布
    GARCH-FHS         GJR-GARCH(1,1) 预测明日波动率 × 标准化残差的经验分位（过滤历史模拟）
    GARCH-EVT         GJR-GARCH(1,1) 预测明日波动率 × 标准化残差尾部的 GPD 分位
                      （McNeil & Frey 2000, Journal of Empirical Finance）

前四种不随波动率变化（无条件）；后两种随 GARCH 预测的波动率每日调整（条件）。
所有预测只用截至当日的数据，用来对次日收益做回测。损失记为正数。

══ 回测 ═════════════════════════════════════════════════════════

    Kupiec（1995）       VaR 突破频率是否等于 1 − 置信水平（无条件覆盖）
    Christoffersen（1998）突破是否扎堆（独立性）
    Acerbi–Szekely（2014）Z2 ES 检验：Z2 = Σ(X_t·I_t)/(T·α·ES_t) + 1，X 为收益；
                         原文给出 α=2.5% 时 5% 水平的临界值约 −0.70（< −0.70 即 ES 被低估）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, genpareto, norm
from scipy.stats import t as student_t



# ══════════════════════════════════════════════════════════════════
# 单个窗口的 VaR / ES（输入收益，返回正的损失）
# ══════════════════════════════════════════════════════════════════

def var_es_hist(r, q):
    loss = -np.asarray(r)
    var = np.quantile(loss, q)
    return var, loss[loss >= var].mean()


def var_es_normal(r, q):
    mu, sd = np.mean(r), np.std(r, ddof=1)
    z = norm.ppf(q)
    return -mu + sd * z, -mu + sd * norm.pdf(z) / (1 - q)


def var_es_t(r, q):
    nu, loc, scale = student_t.fit(r)
    nu = max(nu, 2.05)
    x = student_t.ppf(q, nu)
    es_std = student_t.pdf(x, nu) / (1 - q) * (nu + x ** 2) / (nu - 1)
    return -loc + scale * x, -loc + scale * es_std


def gpd_fit(loss, threshold=0.90):
    """对超过 threshold 分位的损失超额拟合 GPD，返回 (u, ξ, β, 样本数 n, 超额数 N_u)。"""
    loss = np.asarray(loss)
    u = np.quantile(loss, threshold)
    exc = loss[loss > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    return u, xi, beta, len(loss), len(exc)


def gpd_tail(loss, q, threshold=0.90, fit=None):
    """POT-GPD 的 VaR 与 ES（损失为正）。fit 可传入 gpd_fit 的结果以免重复拟合。"""
    u, xi, beta, n, nu = gpd_fit(loss, threshold) if fit is None else fit
    ratio = n / nu * (1 - q)
    if abs(xi) < 1e-6:
        var = u - beta * np.log(ratio)
        return var, var + beta
    var = u + beta / xi * (ratio ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi) if xi < 1 else np.inf
    return var, es


def var_es_gpd(r, q, fit=None):
    return gpd_tail(-np.asarray(r), q, fit=fit)


# ══════════════════════════════════════════════════════════════════
# GJR-GARCH(1,1)：定期重估参数，每日递推波动率
# ══════════════════════════════════════════════════════════════════

def fit_gjr(r):
    """正态 QMLE 拟合 GJR-GARCH(1,1)（收益单位为 %），返回参数 dict。"""
    from arch import arch_model

    res = arch_model(np.asarray(r) * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1,
                     dist="normal").fit(disp="off", show_warning=False)
    p = res.params
    return {"mu": p["mu"], "omega": p["omega"], "alpha": p["alpha[1]"], "gamma": p["gamma[1]"], "beta": p["beta[1]"]}


def gjr_filter(r, params):
    """给定参数，对收益序列递推条件方差；返回 (各日 σ_t, 下一日 σ_{T+1})，单位为收益（非 %）。"""
    x = np.asarray(r) * 100 - params["mu"]
    var = np.empty(len(x) + 1)
    var[0] = x.var()
    for t in range(len(x)):
        neg = x[t] < 0
        var[t + 1] = (params["omega"] + (params["alpha"] + params["gamma"] * neg) * x[t] ** 2
                      + params["beta"] * var[t])
    sd = np.sqrt(var) / 100
    return sd[:-1], sd[-1]


# ══════════════════════════════════════════════════════════════════
# 滚动预测
# ══════════════════════════════════════════════════════════════════

UNCONDITIONAL = {"历史模拟": var_es_hist, "正态": var_es_normal, "t 分布": var_es_t, "GPD": var_es_gpd}


def rolling_forecasts(returns, window=1000, q_var=0.99, q_es=0.975, refit=21, start=None):
    """
    每个交易日收盘后，用过去 window 日的收益预测下一交易日的：
        VaR（q_var，默认 99%）、VaR_es（q_es，默认 97.5%）、ES（q_es）
    ES 回测要用同一置信水平的 VaR，所以另存 VaR_es。
    t 分布与 GARCH 参数每 refit 日重估一次（GARCH 波动率仍每日递推），其余方法每日重算。
    返回 DataFrame：index = 预测所针对的交易日，列为 (方法, 指标)，另有 ("收益", "")。
    """
    r = pd.Series(returns).dropna()
    vals, idx = r.values, r.index
    first = window if start is None else max(window, idx.searchsorted(pd.Timestamp(start)))
    rows, dates = [], []
    t_cache, g_params, z_q = None, None, None
    for i in range(first, len(vals)):
        w = vals[i - window:i]
        row = {}
        for name, f in UNCONDITIONAL.items():
            if name == "t 分布":
                if t_cache is None or (i - first) % refit == 0:
                    t_cache = (f(w, q_var), f(w, q_es))
                (v, _), (ve, e) = t_cache
            elif name == "GPD":
                fit = gpd_fit(-w)
                v, _ = f(w, q_var, fit=fit)
                ve, e = f(w, q_es, fit=fit)
            else:
                v, _ = f(w, q_var)
                ve, e = f(w, q_es)
            row[(name, "VaR")], row[(name, "VaR_es")], row[(name, "ES")] = v, ve, e
        if g_params is None or (i - first) % refit == 0:
            g_params = fit_gjr(w)
            sd_hist, _ = gjr_filter(w, g_params)
            z = (w - g_params["mu"] / 100) / sd_hist
            zl = -z
            zfit = gpd_fit(zl)
            z_q = {"FHS": (var_es_hist(z, q_var)[0], *var_es_hist(z, q_es)),
                   "EVT": (gpd_tail(zl, q_var, fit=zfit)[0], *gpd_tail(zl, q_es, fit=zfit))}
        _, sd_next = gjr_filter(w, g_params)
        mu = g_params["mu"] / 100
        for tag, (zv, zve, ze) in z_q.items():
            row[(f"GARCH-{tag}", "VaR")] = -mu + sd_next * zv
            row[(f"GARCH-{tag}", "VaR_es")] = -mu + sd_next * zve
            row[(f"GARCH-{tag}", "ES")] = -mu + sd_next * ze
        row[("收益", "")] = vals[i]
        rows.append(row)
        dates.append(idx[i])
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


# ══════════════════════════════════════════════════════════════════
# 回测
# ══════════════════════════════════════════════════════════════════

def kupiec(hits, p):
    """无条件覆盖：突破频率是否等于 p。返回 (突破率, p 值)。"""
    hits = np.asarray(hits, bool)
    n, x = len(hits), hits.sum()
    pi = x / n
    if x in (0, n):
        lr = -2 * (x * np.log(p) + (n - x) * np.log(1 - p)) if x == 0 else np.inf
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(pi), float(chi2.sf(lr, 1))


def christoffersen(hits):
    """独立性：今天是否突破与昨天是否突破无关。返回 (突破后再突破的概率, p 值)。"""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00, n01 = ((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum()
    n10, n11 = ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum()
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(k, n, prob):
        return 0.0 if prob in (0, 1) else k * np.log(prob) + (n - k) * np.log(1 - prob)

    lr = -2 * (ll(n01 + n11, n00 + n01 + n10 + n11, p) - ll(n01, n00 + n01, p01) - ll(n11, n10 + n11, p11))
    return float(p11), float(chi2.sf(max(lr, 0.0), 1))


def acerbi_szekely_z2(returns, var, es, alpha):
    """
    Z2 = Σ_t (X_t · I_t) / (T · α · ES_t) + 1，I_t = 1{X_t < −VaR_t}（VaR 与 ES 取同一 α）。
    期望为 0；显著为负说明 ES 被低估。
    """
    x, v, e = (np.asarray(a, float) for a in (returns, var, es))
    hit = x < -v
    return float(np.sum(x * hit / e) / (len(x) * alpha) + 1)


def backtest_table(fc, q_var=0.99, q_es=0.975):
    """
    各方法：99% VaR 的突破率与 Kupiec、Christoffersen 检验；
    97.5% ES 的 Acerbi–Szekely Z2，以及「97.5% VaR 突破日的平均实际损失 / 平均预测 ES」（>1 表示 ES 偏低）。
    """
    x = fc[("收益", "")].values
    methods = [m for m in fc.columns.get_level_values(0).unique() if m != "收益"]
    rows = {}
    for m in methods:
        hit = x < -fc[(m, "VaR")].values
        rate, p_k = kupiec(hit, 1 - q_var)
        p11, p_c = christoffersen(hit)
        ve, es = fc[(m, "VaR_es")].values, fc[(m, "ES")].values
        hit_e = x < -ve
        rows[m] = {"99% VaR 突破率": rate, "Kupiec p": p_k, "突破后次日再突破": p11, "Christoffersen p": p_c,
                   "Z2": acerbi_szekely_z2(x, ve, es, 1 - q_es),
                   "实际损失/预测ES": (-x[hit_e]).mean() / es[hit_e].mean() if hit_e.any() else np.nan}
    return pd.DataFrame(rows).T
