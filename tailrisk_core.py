"""
尾部风险项目 · 核心计算模块

只放**经过验证、不该在 notebook 里重写**的数学：
    - Copula 的极大似然估计（参数边界和独立性检验踩过坑，见下方注释）
    - GARCH 滤波与概率积分变换
    - 同花顺 Excel 导出格式的解析

数据提取、结果展示、图表都放在 notebook 里，方便逐步查看 DataFrame。

依赖：numpy pandas scipy arch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, kendalltau, norm, rankdata

_EPS = 1e-6
LAMBDA_ZERO_TOL = 1e-4


def _clip01(x):
    return np.clip(np.asarray(x, dtype=float), _EPS, 1 - _EPS)


# ══════════════════════════════════════════════════════════════════
# 一、数据解析
# ══════════════════════════════════════════════════════════════════

def parse_ifind_export(path, encoding="utf-8"):
    """
    解析同花顺 Excel/CSV 导出的**多标的横向拼接**格式。

    该格式每个标的占 3 列（日期 / 数值 / 空列），且**各标的的日期列相互独立**
    （不同市场交易日不同，不能假设行对齐）。因此必须逐标的取出
    (日期, 数值) 配对，再按日期外连接，不能直接按行号拼。

    返回：DataFrame，index=日期，每列一个标的，列名为「中文名(代码)」
    """
    with open(path, encoding=encoding) as f:
        lines = f.read().split("\n")

    names, codes = lines[0].split("\t"), lines[1].split("\t")
    body = [l for l in lines[3:] if l.strip()]
    n_inst = len(names) // 3 + (1 if len(names) % 3 else 0)

    out = {}
    for i in range(n_inst):
        dcol, vcol = 3 * i, 3 * i + 1
        if dcol >= len(codes):
            break
        code = codes[dcol].strip()
        name = names[vcol].strip() if vcol < len(names) else ""
        if not code:
            continue
        dates, vals = [], []
        for line in body:
            cells = line.split("\t")
            if vcol >= len(cells):
                continue
            d, v = cells[dcol].strip(), cells[vcol].strip().replace(",", "")
            if not d or not v:
                continue
            try:
                vals.append(float(v))
                dates.append(pd.Timestamp(d))
            except ValueError:
                continue
        if dates:
            out[f"{name}({code})"] = pd.Series(vals, index=dates)

    if not out:
        return pd.DataFrame()
    return pd.concat(out.values(), axis=1, keys=out.keys()).sort_index()


def data_quality(df):
    """
    数据质量体检。返回 DataFrame，一行一个标的。

    关注 ties（零收益率、重复值）：PIT 建立在经验分布上，大量重复值会让
    多个观测映射到同一分位，Copula 的似然曲面趋于平坦，MLE 系统性偏向
    独立解 —— 表现为 lambda 恒为 0，与「真的没有尾部相关」无法区分。
    """
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) < 3:
            rows.append({"标的": c, "样本数": len(s), "可用": False})
            continue
        r = s.pct_change().dropna()
        n_zero = int((r.abs() < 1e-10).sum())
        rows.append({
            "标的": c,
            "样本数": len(s),
            "起点": s.index.min().date(),
            "终点": s.index.max().date(),
            "年化收益": (s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1,
            "年化波动": r.std() * np.sqrt(244),
            "最大回撤": (s / s.cummax() - 1).min(),
            "零收益占比": n_zero / len(r),
            "唯一值占比": r.nunique() / len(r),
            "可用": (n_zero / len(r) < 0.05) and (r.nunique() / len(r) > 0.80),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# 二、边缘分布与 PIT
# ══════════════════════════════════════════════════════════════════

def pit_empirical(r):
    """
    经验 CDF 变换（秩变换）。最贴近中金原文字面 —— 原文未提任何滤波。

    除以 (n+1) 保证结果落在开区间 (0,1)，避免 Copula 对数似然出现 log(0)。
    ties 用 average 处理。
    """
    s = pd.Series(r).dropna()
    return pd.Series(rankdata(s.values, method="average") / (len(s) + 1), index=s.index)


def pit_garch(r, dist="skewt"):
    """
    GJR-GARCH 滤波 + 标准化残差的经验 PIT。

    ⚠ 这不是中金原文的做法，是本项目的对照版本。
    危机期「股债同步暴跌」在数据上很大程度表现为两者波动率同时放大，
    GARCH 滤波恰好剥离这一层，可能把要测的东西一起剥掉。
    模拟检验显示：残差独立、仅波动率由共同因子驱动的数据，
    滤波前 lambda_D>0 占 6.4%，滤波后掉到 2.2%（等于噪声基准）。
    两个版本要并排出结果。
    """
    from arch import arch_model

    s = pd.Series(r).dropna()
    am = arch_model(s * 100, vol="GARCH", p=1, o=1, q=1, dist=dist)
    res = am.fit(disp="off", show_warning=False)
    z = res.std_resid.dropna()
    return pd.Series(rankdata(z.values, method="average") / (len(z) + 1), index=z.index)


def pit_vol_scaled(r, window=60):
    """
    滚动波动率标准化后再做经验 PIT。介于上面两者之间的中间档。

    用 shift(1)：标准化只能用截至前一日的波动率，否则当日收益参与了
    自己的标准化，引入前视偏差。
    """
    s = pd.Series(r).dropna()
    vol = s.rolling(window, min_periods=window // 2).std().shift(1)
    z = (s / vol).dropna()
    return pd.Series(rankdata(z.values, method="average") / (len(z) + 1), index=z.index)


# ══════════════════════════════════════════════════════════════════
# 三、Copula
# ══════════════════════════════════════════════════════════════════

class ClaytonCopula:
    """
    仅下尾相关：lambda_D = 2^(-1/theta)，theta→0 对应独立。

    ⚠ 独立性似然比检验是必须的。Clayton 是「强制拟合」的 —— 即使数据完全
    独立，MLE 也会给出一个小的正 theta，算出虚假的正 lambda_D。
    早期版本因优化下界卡在 1e-3，纯高斯数据也能测出 lambda_D 峰值 0.74，
    使得中金原文「lambda_D>0」的判别条件恒为真、完全失效。

    检验：H0 为独立（theta=0，copula 密度恒为1，对数似然为0）。
    若 2*loglik 未超过 chi2(1) 临界值，判定 lambda_D = 0。

    注：theta=0 在参数空间边界上，严格的零分布是 chi2(0) 与 chi2(1) 的
    50:50 混合，用纯 chi2(1) 偏保守（实测独立数据的误报率 2.0% < 名义 5%）。
    保守方向，暂不修正。
    """

    name = "clayton"

    def _neg_loglik(self, theta, u, v):
        if theta <= 1e-4:
            return 1e10
        ll = (np.log(1 + theta)
              - (theta + 1) * (np.log(u) + np.log(v))
              - (1 / theta + 2) * np.log(u ** (-theta) + v ** (-theta) - 1))
        return -np.sum(ll)

    def fit(self, u, v, indep_alpha=0.05):
        u, v = _clip01(u), _clip01(v)
        res = minimize_scalar(self._neg_loglik, args=(u, v),
                              bounds=(1e-6, 50), method="bounded")
        self.theta = float(res.x)
        self.loglik = -res.fun
        self.indep_pvalue = float(chi2.sf(2 * max(self.loglik, 0.0), df=1))
        self.is_independent = self.indep_pvalue > indep_alpha
        return self

    @property
    def lambda_lower(self):
        if self.is_independent:
            return 0.0
        return float(2 ** (-1 / self.theta)) if self.theta > 1e-6 else 0.0


class GumbelCopula:
    """
    仅上尾相关：lambda_U = 2 - 2^(1/theta)，theta→1 对应独立。

    ⚠ 已知局限：独立性检验的原假设是「独立」而非「无上尾相关」。
    有相关的数据本来就不独立，检验必然拒绝，于是给出正的 lambda_U。
    模拟显示：即使数据是纯 Clayton（真实 lambda_U 严格为 0），
    Gumbel 仍在 100% 的窗口报出 lambda_U>0，中金条件命中率为 0。

    这意味着「lambda_U=0」这一条在相关性较高时可能永不成立。
    备选判定（ll_Clayton - ll_Gumbel > 阈值）判别力更干净：
    真下尾 98.7% 命中、真上尾 0%、对称情形 9-18% 假阳性。
    两者并排输出，见 rolling_tail_dependence 的 ll_diff 列。
    """

    name = "gumbel"

    def _neg_loglik(self, theta, x, y):
        if theta < 1.0:
            return 1e10
        A = x ** theta + y ** theta
        log_c = (-(A ** (1 / theta)) + x + y
                 + (theta - 1) * (np.log(x) + np.log(y))
                 + (1 / theta - 2) * np.log(A)
                 + np.log(theta - 1 + A ** (1 / theta)))
        return -np.sum(log_c)

    def fit(self, u, v, indep_alpha=0.05):
        u, v = _clip01(u), _clip01(v)
        x, y = -np.log(u), -np.log(v)
        res = minimize_scalar(self._neg_loglik, args=(x, y),
                              bounds=(1.0, 30), method="bounded")
        self.theta = float(res.x)
        self.loglik = -res.fun
        self.indep_pvalue = float(chi2.sf(2 * max(self.loglik, 0.0), df=1))
        self.is_independent = self.indep_pvalue > indep_alpha
        return self

    @property
    def lambda_upper(self):
        if self.is_independent:
            return 0.0
        return float(2 - 2 ** (1 / self.theta)) if self.theta > 1.0 else 0.0


class GaussianCopula:
    """基准模型，尾部相关系数恒为 0 —— 这正是它低估尾部风险的原因。"""

    name = "gaussian"

    def fit(self, u, v):
        u, v = _clip01(u), _clip01(v)
        x, y = norm.ppf(u), norm.ppf(v)
        self.rho = float(np.corrcoef(x, y)[0, 1])
        det = 1 - self.rho ** 2
        self.loglik = float(np.sum(
            -0.5 * np.log(det)
            - (self.rho ** 2 * (x ** 2 + y ** 2) - 2 * self.rho * x * y) / (2 * det)))
        return self

    lambda_lower = 0.0
    lambda_upper = 0.0


# ══════════════════════════════════════════════════════════════════
# 四、滚动估计
# ══════════════════════════════════════════════════════════════════

def rolling_tail_dependence(u, v, window=60, step=1):
    """
    滚动窗口内分别拟合 Clayton 与 Gumbel，各取 lambda_D / lambda_U。
    中金原文口径：两者独立估计，不做模型选择。

    返回 DataFrame，列：
        lambda_D, lambda_U          尾部相关系数
        clayton_theta, gumbel_theta 估计参数
        clayton_ll, gumbel_ll       对数似然
        ll_diff                     ll_Clayton - ll_Gumbel（尾部不对称诊断用）
    """
    u, v = pd.Series(u).dropna(), pd.Series(v).dropna()
    idx = u.index.intersection(v.index)
    u, v = u.reindex(idx), v.reindex(idx)

    rows = []
    for end in range(window, len(idx) + 1, step):
        uu = u.iloc[end - window:end].values
        vv = v.iloc[end - window:end].values
        try:
            c = ClaytonCopula().fit(uu, vv)
            ld, th_c, ll_c = c.lambda_lower, c.theta, c.loglik
        except Exception:
            ld, th_c, ll_c = np.nan, np.nan, np.nan
        try:
            g = GumbelCopula().fit(uu, vv)
            lu, th_g, ll_g = g.lambda_upper, g.theta, g.loglik
        except Exception:
            lu, th_g, ll_g = np.nan, np.nan, np.nan
        rows.append({
            "date": idx[end - 1],
            "lambda_D": ld, "lambda_U": lu,
            "clayton_theta": th_c, "gumbel_theta": th_g,
            "clayton_ll": ll_c, "gumbel_ll": ll_g,
            "ll_diff": ll_c - ll_g if np.isfinite(ll_c) and np.isfinite(ll_g) else np.nan,
        })
    return pd.DataFrame(rows).set_index("date")


def dif_trend(px, fast=12, slow=26):
    """标准 DIF = MA(12) - MA(26)。>0 上涨趋势，<0 下跌趋势。"""
    s = pd.Series(px).dropna()
    return s.rolling(fast).mean() - s.rolling(slow).mean()


# ══════════════════════════════════════════════════════════════════
# 五、噪声基准
# ══════════════════════════════════════════════════════════════════

def noise_baseline(window=60, n_rep=300, rho=0.0, seed=1):
    """
    生成「独立/无尾部相关」数据下同一统计量的取值，作为对照基准。

    这是判断「测出了信号」还是「测出了噪声」的唯一依据。
    首轮中金复现之所以能定性为「信号未产生」而非「信号很弱」，
    靠的就是这个对照：实测 lambda_D>0 占比 2.0%，
    恰好等于纯独立数据的 2.0%。

    返回 dict：lambda_D>0 占比、lambda_U>0 占比、中金条件命中率
    """
    rng = np.random.default_rng(seed)
    lds, lus = [], []
    for _ in range(n_rep):
        z1 = rng.standard_normal(window)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(window)
        u, v = norm.cdf(z1), norm.cdf(z2)
        lds.append(ClaytonCopula().fit(u, v).lambda_lower)
        lus.append(GumbelCopula().fit(u, v).lambda_upper)
    lds, lus = np.array(lds), np.array(lus)
    return {
        "rho": rho,
        "lambda_D>0占比": (lds > LAMBDA_ZERO_TOL).mean(),
        "lambda_U>0占比": (lus > LAMBDA_ZERO_TOL).mean(),
        "中金条件命中率": ((lds > LAMBDA_ZERO_TOL) & (lus <= LAMBDA_ZERO_TOL)).mean(),
        "lambda_D均值": lds.mean(),
    }
