# 尾部风险复现 · Notebook 版

## 文件

| 文件 | 内容 |
|---|---|
| `tailrisk_core.py` | 核心数学：Copula 估计、PIT、同花顺导出解析、噪声基准 |
| `01_L0_L2_数据与尾部相关性.ipynb` | 数据提取 → 质量体检 → 边缘分布 → 滚动尾部相关系数 |
| `02_中金_股债双杀预警复现.ipynb` | 趋势条件 → 预警状态机 → 情景对照表 → 回测 |
| `data/` | 放同花顺导出的 txt（制表符分隔） |
| `output/` | 结果导出 |

## 环境

```bash
pip install numpy pandas scipy arch matplotlib jupyter jqdatasdk
```

`arch` 只被 `pit_garch()` 用到，且是函数内 import，装不上也不影响其余部分
（主口径 `pit_empirical` 才是贴合中金原文的做法）。

## 聚宽凭据

三种方式任选，notebook 会按 环境变量 → credentials.py → 交互输入 依次尝试。

**方式一：credentials.py（推荐，与启动方式无关）**

```bash
cp credentials_example.py credentials.py   # 然后填入账号密码
```

已在 .gitignore 中，不会进版本库。

**方式二：环境变量**

```bash
export JQ_USERNAME=... JQ_PASSWORD=...
jupyter notebook          # 必须从这个终端启动
```

⚠ 只对**从该终端启动**的进程有效。从 Dock/Spotlight 点开的 VS Code
拿不到，其内核也拿不到；若要用 VS Code，需 `code .` 从终端启动。

**方式三**：什么都不配置，运行时会提示输入，密码不回显。

## 常见问题

**包装了却报 ModuleNotFoundError** —— VS Code 的内核和 pip 装包的 Python
不是同一个。在 notebook 里跑 `import sys; print(sys.executable)` 确认，
然后用 `!{sys.executable} -m pip install ...` 装到当前内核对应的环境。

**图表中文显示成方框** —— 跑下面这段看系统有哪些中文字体，
把结果填进 notebook 第 0 格的字体列表：

```python
from matplotlib import font_manager
print([f.name for f in font_manager.fontManager.ttflist
       if any(k in f.name for k in ["Hei", "PingFang", "Song", "Kai"])])
```

## 结构约定

每个 notebook 都是**先提取数据落成 DataFrame，确认无误再往下算**。
核心数学放在 `tailrisk_core.py`（Copula 参数边界和独立性检验踩过坑，
有回归测试背书，不在 notebook 里重写）。

## 换债券标的

`02` 第 0 节改 `BOND` 一行即可，其余代码不动：

- `上证国债指数`（2005–今，纯国债）
- `上证企业债指数`（2005–2020，含信用，ties 最干净）
- `中债综合财富指数`（2021–今，口径最贴原文但只有 5 年）

## 已知的方法局限

**λ_U 判零条件。** 独立性检验的原假设是「独立」而非「无上尾相关」。
模拟显示：即使数据是纯 Clayton（真实 λ_U 严格为 0），Gumbel 仍在 100%
的窗口报出 λ_U>0。相关性较高时中金的「λ_U=0」可能永不成立。
备选判定 `ll_diff = ll_Clayton − ll_Gumbel` 已一并输出。

**GARCH 滤波。** 会剥离波动率联动，而危机期股债同步暴跌很大程度就表现为
两者波动率同时放大。主口径用 `pit_empirical`（贴合原文），
`pit_garch` 作为对照版本并排出结果。

## 纪律

- 任何「测出了信号」的结论必须对照噪声基准（`tc.noise_baseline`）
- 所有滚动分位数用扩展窗口或 shift(1)，避免前视偏差
- 不得调参数直到情景③变负
