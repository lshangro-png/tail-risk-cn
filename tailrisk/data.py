"""
数据层：取数、本地缓存、导出文件解析、质量体检。

主数据源免账号：中债指数直接请求中国债券信息网，沪深300 等经 AkShare 取自新浪，
均可回溯到 2002 年，覆盖中金原文的全部样本期。
聚宽作为备选，需要账号（见 credentials_example.py）。

⚠ 取到的数据只缓存在 data/cache/（已 gitignore），不得提交到版本库：
  中债、聚宽、万得的数据都不允许再分发，仓库里只放取数代码。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# 名称 → (来源, 参数)。新增标的只需在这里加一行。
SERIES = {
    "沪深300":         ("ak_index", "sh000300"),
    "中证500":         ("ak_index", "sh000905"),
    "上证国债指数":     ("ak_index", "sh000012"),
    "上证企业债指数":   ("ak_index", "sh000013"),
    "中债综合财富":     ("chinabond", "2c90818811afed8d0111c0c672b31578"),  # 中金原文所用
    "中债新综合财富":   ("chinabond", "8a8b2ca0332abed20134ea76d8885831"),
    "中债国债总财富":   ("chinabond", "2c9081e50e8767dc010e879acb220021"),  # 国泰君安（2018）所用
    "黄金Au9999":      ("ak_sge", "Au99.99"),       # 上海金交所现货，免费数据仅 2016-12 起
    "黄金ETF":         ("ak_etf_hfq", "518880"),     # 华安黄金ETF（跟踪 Au99.99，后复权），2013-07 起；与 A 股同时段交易
    # 东北证券（2025）示例中的 10 只 ETF，均为后复权（含分红再投资）
    "沪深300ETF":      ("ak_etf_hfq", "510300"),
    "中证500ETF":      ("ak_etf_hfq", "159922"),
    "中证1000ETF":     ("ak_etf_hfq", "512100"),
    "创业板ETF":       ("ak_etf_hfq", "159908"),
    "消费ETF":         ("ak_etf_hfq", "159928"),
    "纳指ETF":         ("ak_etf_hfq", "513100"),
    "恒生ETF":         ("ak_etf_hfq", "159920"),
    "红利ETF":         ("ak_etf_hfq", "510880"),
    "大宗商品ETF":     ("ak_etf_hfq", "510170"),
    "标普500ETF":      ("ak_etf_hfq", "513500"),     # 国泰君安（2018）五资产示例中标普500 的同步替代
    # 国泰君安（2018）复现用的长历史序列（2006 年起）
    "标普500":         ("ak_us_index", ".INX"),
    "恒生指数(盈富基金)": ("ak_hk_hfq", "02800"),       # 新浪恒指仅 2013 年起；盈富基金与恒指日收益相关 0.98
    "黄金(GLD×美元兑人民币)": ("gold_cny_gld", None),  # 免费的上金所 Au99.99 仅 2016-12 起
    # 兴业（2025）单资产尾部预警所需
    "GLD":             ("ak_us_daily", "GLD"),
    "纳斯达克100":      ("fred", "NASDAQ100"),
    "布伦特原油":       ("fred", "DCOILBRENTEU"),
    "美国10年实际利率":  ("fred", "DFII10"),
    "VIX":             ("fred", "VIXCLS"),
    "VXN":             ("fred", "VXNCLS"),
    "OVX":             ("fred", "OVXCLS"),
    "美元指数(美联储发达经济体)": ("fred", "DTWEXAFEGS"),   # ICE 美元指数无免费数据，此为构成接近的替代
    "美元兑人民币":     ("fred", "DEXCHUS"),               # 离岸人民币无免费长历史，用在岸替代
    "上证A股股息率":    ("lg_gxl", "上证A股"),
    "中国10年国债收益率": ("cn_10y", None),
}

# 申万一级行业（31 个），用于以行业代替成分股计算 PcaScore
SW_L1 = ["801010", "801030", "801040", "801050", "801080", "801110", "801120", "801130", "801140",
         "801150", "801160", "801170", "801180", "801200", "801210", "801230", "801710", "801720",
         "801730", "801740", "801750", "801760", "801770", "801780", "801790", "801880", "801890",
         "801950", "801960", "801970", "801980"]

ETF10 = ["沪深300ETF", "中证500ETF", "中证1000ETF", "创业板ETF", "消费ETF",
         "黄金ETF", "纳指ETF", "恒生ETF", "红利ETF", "大宗商品ETF"]


_CHINABOND_URL = "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQuery"


def _fetch_chinabond(index_id):
    """
    中国债券信息网单指数查询（财富指数、总值）。

    不用 AkShare 的 bond_composite_index_cbond：它把毫秒时间戳按 UTC 转日期，
    北京时间零点变成前一天，整条序列错位一天（与股票收益的相关性因此被打乱）。
    这里按 Asia/Shanghai 转换。
    """
    import requests

    params = {"indexid": index_id, "qxlxt": "00", "zslxt": "CFZS", "lx": "1", "locale": ""}
    js = requests.post(_CHINABOND_URL, params=params, timeout=30).json()["CFZS_00"]
    ts = pd.to_datetime(pd.Index(js.keys()).astype(float), unit="ms", utc=True)
    idx = ts.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    return pd.Series(pd.to_numeric(list(js.values()), errors="coerce"), index=idx)


_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _fetch_fred(series_id):
    """圣路易斯联储 FRED 公开数据（CSV，无需账号）。"""
    df = pd.read_csv(_FRED_CSV.format(series_id), index_col=0, parse_dates=True)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()


def _fetch(source, arg):
    if source == "chinabond":
        return _fetch_chinabond(arg)
    if source == "fred":
        return _fetch_fred(arg)
    if source == "gold_cny_gld":
        # SPDR 黄金 ETF（美元）× 美元兑人民币（FRED DEXCHUS），按美股交易日对齐、汇率前向填充
        gld = _fetch("ak_us_daily", "GLD")
        fx = _fetch_fred("DEXCHUS").reindex(gld.index, method="ffill")
        return (gld * fx).dropna()

    import akshare as ak

    if source == "ak_index":
        df = ak.stock_zh_index_daily(symbol=arg)
        return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    if source == "ak_etf_hfq":
        # 东方财富后复权；该接口连续请求易被限流，失败时改用腾讯后复权
        # （2016 年以来与东方财富的日收益差异 < 0.08%，来自价格三位小数的舍入）
        try:
            df = ak.fund_etf_hist_em(symbol=arg, period="daily", adjust="hfq")
            return pd.Series(df["收盘"].values, index=pd.to_datetime(df["日期"]))
        except Exception:
            prefix = "sh" if arg.startswith("5") else "sz"
            df = ak.stock_zh_a_hist_tx(symbol=prefix + arg, start_date="20000101",
                                       end_date=pd.Timestamp.today().strftime("%Y%m%d"), adjust="hfq")
            return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    if source == "lg_gxl":
        df = ak.stock_a_gxl_lg(symbol=arg)
        return pd.Series(df["股息率"].values, index=pd.to_datetime(df["日期"]))
    if source == "cn_10y":
        df = ak.bond_zh_us_rate(start_date="20020101")
        return pd.Series(df["中国国债收益率10年"].values, index=pd.to_datetime(df["日期"])).dropna()
    if source == "sw_index":
        df = ak.index_hist_sw(symbol=arg, period="day")
        return pd.Series(df["收盘"].values, index=pd.to_datetime(df["日期"]))
    if source == "ak_us_index":
        df = ak.index_us_stock_sina(symbol=arg)
        return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    if source == "ak_us_daily":
        df = ak.stock_us_daily(symbol=arg, adjust="")
        return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    if source == "ak_hk_hfq":
        df = ak.stock_hk_daily(symbol=arg, adjust="hfq")
        return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    if source == "ak_sge":
        df = ak.spot_hist_sge(symbol=arg)
        return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    raise ValueError(f"未知数据源 {source}")


def _fetch_with_retry(source, arg, tries=4):
    """部分免费接口（如东方财富）连续请求过快会断开连接：间隔递增地重试几次。"""
    import time

    for k in range(tries):
        try:
            return _fetch(source, arg)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(10 * 2 ** k)


def load_series(name, refresh=False):
    """
    按名称取一条日频价格序列（收盘价 / 财富指数值），优先读本地缓存。

    缓存为 data/cache/<名称>.csv。refresh=True 时强制重新下载。
    """
    if name not in SERIES:
        raise KeyError(f"{name} 不在 SERIES 中，可选：{list(SERIES)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.csv"
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        s = _fetch_with_retry(*SERIES[name]).sort_index()
        s = s[~s.index.duplicated(keep="last")].astype(float)
        s.to_frame(name).to_csv(path)
    s.name = name
    return s.dropna()


def load_prices(names, start=None, end=None, how="inner", refresh=False):
    """
    取多条序列并按日期对齐。默认 inner：只保留共同交易日。

    不做前向填充 —— 填充会制造零收益（ties），让 Copula 的 MLE 偏向独立解。
    """
    df = pd.concat([load_series(n, refresh=refresh) for n in names], axis=1, join=how)
    return df.loc[start:end].sort_index()


def load_sw_industries(refresh=False):
    """申万一级行业收盘点位（31 列），各自缓存为 data/cache/申万_<代码>.csv。"""
    import time

    out = {}
    for code in SW_L1:
        name = f"申万_{code}"
        if name not in SERIES:
            SERIES[name] = ("sw_index", code)
        cached = (CACHE_DIR / f"{name}.csv").exists()
        out[code] = load_series(name, refresh=refresh)
        if not cached:
            time.sleep(1)
    return pd.concat(out, axis=1).sort_index()


def to_returns(prices):
    """
    简单收益率。显式 fill_method=None：pandas 的 pct_change 默认会对 NaN
    做前向填充，缺失日变成 0 收益、下一日跨缺口计收益，恰好制造 ties。
    """
    return prices.pct_change(fill_method=None).dropna(how="all")


# ══════════════════════════════════════════════════════════════════
# 聚宽（备选来源，需账号）
# ══════════════════════════════════════════════════════════════════

def jq_login():
    """
    聚宽登录，三级回退：环境变量 → credentials.py → 交互输入。

    终端里 export 的变量只有从该终端启动的进程才继承得到；
    从 Dock/Spotlight 打开的 VS Code 及其内核拿不到，所以需要回退。
    """
    import jqdatasdk as jq

    u, p = os.environ.get("JQ_USERNAME"), os.environ.get("JQ_PASSWORD")
    src = "环境变量"
    if not (u and p):
        try:
            import credentials
            u, p = credentials.JQ_USERNAME, credentials.JQ_PASSWORD
            src = "credentials.py"
        except (ImportError, AttributeError):
            from getpass import getpass
            u, p = input("聚宽账号: "), getpass("密码: ")
            src = "交互输入"
    jq.auth(u, p)
    print(f"聚宽登录成功（凭据来自{src}）")
    return jq


# ══════════════════════════════════════════════════════════════════
# 同花顺导出解析
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


# ══════════════════════════════════════════════════════════════════
# 质量体检
# ══════════════════════════════════════════════════════════════════

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
