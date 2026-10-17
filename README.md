# 尾部风险：研报复现与方法检验

本项目复现国内外关于尾部风险识别、资产配置与预警的研究，并检验每篇研究的核心结论：
它测到的是信号，还是噪声？

各篇复现共用一套核心库 `tailrisk/`，每篇研究放在 `studies/` 下的独立目录，
目录里的 README 按固定格式写明：原文结论 → 口径差异 → 复现结果 → 判定。

## 研究进度

| 目录 | 研究 | 状态 | 判定 |
|---|---|---|---|
| `studies/01_single_asset_tail/` | 单资产肥尾、GPD、ES/VaR | 计划中 | |
| [`studies/02_tail_dependence/`](studies/02_tail_dependence/) | 尾部相关性：数据、PIT、Copula、噪声基准 | ✅ | 方法基础设施 |
| [`studies/03_cicc_stock_bond_warning/`](studies/03_cicc_stock_bond_warning/) | 中金（2021）股债双杀预警 | ✅ 股债 + 股债金多资产 + 延伸检验（相关系数对照、样本外、t、SJC） | **部分复现，核心证据不成立**（样本外同样不成立） |
| `studies/04_cib_single_asset_warning/` | 兴业（2025）单资产尾部预警 | 计划中 | |
| [`studies/05_tail_hierarchical_allocation/`](studies/05_tail_hierarchical_allocation/) | 基于尾部相关的分层聚类配置（源自整理稿记载的东北证券方法，未取得原文） | ✅ | **尾部相关在聚类、合成、配置上都无增量** |
| `studies/06_gtja_kde_simulation/` | 国泰君安 KDE + Cholesky 模拟配置 | 计划中 | |
| `studies/07_tail_risk_parity/` | 尾部风险平价（历史 ETL 版） | 计划中 | |

## 目录结构

```
tailrisk/          核心库
  data.py            取数（中债官网 / AkShare / 聚宽）、缓存、同花顺导出解析、质量体检
  marginals.py       边缘分布与 PIT、单资产 ES（历史法 / GPD）
  copula.py          Clayton / Gumbel / Gaussian、滚动尾部相关、噪声基准
  elliptical.py      t-Copula 与高斯的似然比检验（是否有超出普通相关的尾部联动）
  sjc.py             SJC Copula：上下尾同时估计、不对称检验、旋转（股跌债涨）
  allocation.py      距离与聚类、HRP、最小方差、最小 ES、arctan 递归合成、滚动回测
  signals.py         趋势、预警状态机、事件核对
  backtest.py        日频回测、风险平价、绩效统计
studies/           各篇复现，每篇一个目录（notebook + README）
tests/             回归测试
data/cache/        本地数据缓存（不入库）
```

## 快速开始

```bash
pip install -e ".[dev,garch]"
pytest -q
jupyter notebook studies/03_cicc_stock_bond_warning/stock_bond_warning.ipynb
```

数据首次运行时自动下载，缓存在 `data/cache/`。中债指数直接请求中国债券信息网，
股票指数经 AkShare 取自新浪，**都不需要账号**。
聚宽是备选来源，用法见 `credentials_example.py`：复制为 `credentials.py` 后填写，该文件已加入 gitignore。

**数据不入库**：中债、聚宽、万得的数据都不允许再分发，仓库里只提交取数代码。

## 贯穿全项目的方法结论

1. **噪声基准必须匹配相关强度。** Clayton/Gumbel 的 λ>0 主要反映窗口内是否正相关，
   与有没有尾部相关关系不大。所以对照基准要用「相关强度相同、尾部相关为 0」的高斯 copula
   （`copula.expected_under_gaussian`），只和独立数据比是不够的。
2. **短窗口下的单参数 Copula 测到的主要是普通相关。** 60 日窗口的 Clayton/Gumbel λ 几乎完全由窗口相关系数决定（AUC 0.95）。
   要检验「超出普通相关的尾部联动」，应该用有独立尾部参数的模型（如 t 对比高斯，`elliptical.t_vs_gaussian`），并且给足样本。
   要检验上下尾是否不对称，应该在同一个模型里同时估计两尾（SJC，`sjc.asymmetry_test`），
   而不是像中金那样用 Clayton、Gumbel 各估各的。尾部相关的**水平**仍混有普通相关，要和同等相关强度的高斯基准对照
   （`sjc.gaussian_tau_baseline`）。在 10 只 ETF 上，Clayton λ_L 与相关系数的秩相关约 0.98，
   用它代替相关系数做聚类、合成组合风险或配置，都没有带来增量（见 `studies/05`）。
3. **边缘分布只用窗口内的数据。** 全样本 PIT 有前视偏差，还会把波动率聚集误判成尾部相关。
4. **判零口径要写清楚。** 点估计判零和 LR 检验判零的结论差别很大，两种都要报告。

## 纪律

- 凡是「测出了信号」的结论，都必须对照噪声基准
- 所有滚动统计只能用截至当日的数据；信号在 t 日收盘后确定，最早在 t+1 日交易
- 不为了对上原文去调参数。原文没写清的地方，在 README 的口径差异表里列出，并做稳健性检验
