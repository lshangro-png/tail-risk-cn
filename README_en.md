# Tail Risk in Chinese Multi-Asset Portfolios: Replications and Method Tests

[中文](README.md)

This project replicates sell-side research on tail-risk measurement, allocation and early-warning signals
(CICC, Guotai Junan, Industrial Securities, Northeast Securities, and AllianceBernstein's tail risk parity).
For each study it asks the same question: **does the method pick up a real signal, or noise?**

All studies share one library, `tailrisk/`. Each study lives in its own folder under `studies/`. The folder's README
follows a fixed template: original claim → differences in setup → replication results → verdict.
The notebooks and study READMEs are written in Chinese.

## Studies

| Folder | Study | Verdict |
|---|---|---|
| [`01_single_asset_tail`](studies/01_single_asset_tail/) | Fat-tail facts; rolling out-of-sample backtests of six VaR/ES estimators on 7 assets | Fat-tail facts hold. **GARCH-EVT** (McNeil & Frey 2000) is the most reliable; the normal model underestimates ES by about 27% |
| [`02_tail_dependence`](studies/02_tail_dependence/) | Infrastructure: data, PIT, copulas, noise baselines | — |
| [`03_cicc_stock_bond_warning`](studies/03_cicc_stock_bond_warning/) | CICC (2021): stock-bond "double-kill" warning from rolling Clayton/Gumbel tail dependence; stock/bond/gold extension | The computation replicates (3 of 5 warnings within ±1 day), but the **core evidence does not**. λ is essentially a monotone function of the rolling correlation (AUC 0.95). The multi-asset result rests almost entirely on 2008. The signal also fails out of sample |
| [`04_cib_single_asset_warning`](studies/04_cib_single_asset_warning/) | Industrial Securities (2025): single-asset tail warnings (VIX, real yields, OVX, PCA herding) | 6 of 7 assets replicate, but **each effect hinges on a single crisis**. Dropping the largest episode removes it (2015, 2020-03 or 2022). The gold signal has look-ahead bias from US/Asia close timing |
| [`05_tail_hierarchical_allocation`](studies/05_tail_hierarchical_allocation/) | Hierarchical clustering on lower-tail dependence (method as described for Northeast Securities; original not available) | Tail dependence adds **nothing** over plain correlation in clustering, risk aggregation or out-of-sample allocation |
| [`06_gtja_kde_simulation`](studies/06_gtja_kde_simulation/) | Guotai Junan (2018): KDE simulation and minimisation of an inner-/end-of-horizon risk measure | 2006–2017 results largely replicate. The Cholesky recursion distorts variances and marginals and depends on asset ordering, but has little effect on results here. Performance deteriorates after publication |
| [`07_tail_risk_parity`](studies/07_tail_risk_parity/) | Tail risk parity (ES-budgeting) vs. risk parity | Slightly better but not significant. **The gain comes from GARCH volatility scaling, not from using ES** |

## Cross-cutting findings

1. **Noise baselines must match the correlation level.** With 60-day windows, single-parameter Clayton/Gumbel tail coefficients
   are driven almost entirely by the ordinary correlation inside the window. Benchmark them against a Gaussian copula with the
   same correlation and zero tail dependence (`copula.expected_under_gaussian`), not against independence.
2. **Test tail dependence with models that have separate tail parameters**: t-copula vs. Gaussian likelihood-ratio tests
   (`elliptical.t_vs_gaussian`) and the SJC copula (`sjc.asymmetry_test`).
   On CSI 300 × ChinaBond, there is no evidence that joint crashes are stronger than joint rallies.
3. **What actually helps is tracking current volatility and monitoring correlation.** Replacing volatility with ES, or
   correlation with tail dependence, adds little on these data.
4. **Count independent episodes.** Daily-averaged event studies often rest on one or two crises.
5. **Align time zones.** US data released after the Asian close must be lagged one day when it is used to signal Asian assets.

## Layout

```
tailrisk/          shared library
  data.py            data sources (ChinaBond, AkShare, FRED, JoinQuant), caching, quality checks
  marginals.py       PIT; historical and GPD expected shortfall
  riskmodels.py      rolling VaR/ES (HS, normal, t, GPD, GARCH-FHS, GARCH-EVT); Kupiec, Christoffersen, Acerbi–Szekely tests
  copula.py          Clayton / Gumbel / Gaussian, rolling tail dependence, noise baselines
  elliptical.py      t-copula vs. Gaussian LR test
  sjc.py             symmetrised Joe-Clayton copula, asymmetry test
  allocation.py      HRP, min-variance, min-CVaR, risk parity and ES parity (convex risk budgeting), backtests
  simulation.py      KDE marginals with three dependence schemes (GTJA recursion, Gaussian copula, Iman-Conover)
  signals.py         trend, warning state machine
  warning.py         single-asset warnings, event studies, placebo and leave-one-episode-out tests
  backtest.py        daily backtests and performance statistics
studies/           one folder per study (notebook + README)
tests/             regression tests
```

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
jupyter notebook studies/01_single_asset_tail/single_asset_tail.ipynb
```

Data is downloaded on first use and cached in `data/cache/`. All default sources are free and need no account:
ChinaBond, AkShare (Sina, Tencent and Eastmoney), and FRED.
**No data is committed**: vendor data (ChinaBond, Wind, JoinQuant) may not be redistributed, so the repository only contains
the code that fetches it.

## Ground rules

- Every "we found a signal" claim is checked against a noise baseline.
- Rolling statistics use only data available at the time. A signal formed at day-*t* close trades at *t+1* at the earliest.
- No tuning parameters to match the original paper. Unspecified details are listed in each study's README
  and tested for robustness.
