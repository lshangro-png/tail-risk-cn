"""
尾部风险研究共享库。

    data        取数、缓存、导出文件解析、质量体检
    marginals   边缘分布与 PIT
    copula      Copula 估计、滚动尾部相关、噪声基准
    signals     趋势、预警状态机、事件核对
    backtest    日频回测与绩效统计
"""

from . import backtest, copula, data, marginals, signals  # noqa: F401
