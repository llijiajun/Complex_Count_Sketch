"""Build notebooks/paper_tables.ipynb (unexecuted) for the CCS artifact.

The notebook displays each table script's source and then runs it live, so
the delivered file carries real executed outputs.  Execution happens in a
separate step (execute_notebook.py) via nbclient.
"""
import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip() + "\n"))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip() + "\n"))


md(r"""
# 论文表格对账本 — CCS（Complex Count Sketch）

本 notebook 把论文四张表格（Table 2-5）与两张图（fig_sweep / fig_regime）的
实验代码集中在一处，**先展示脚本源码，再原样运行**，运行输出即论文值
facsimile：每格并列"复现值 / 发表值"，命中标注 `*`、偏离标注 `!`，可逐格
与论文核对。

- 脚本本体在 `../tables/`，本 notebook 通过 `inspect.getsource` 展示并调用，
  不会与脚本产生漂移。
- 所有数字均为 **2026-10-23 在 py311 / numpy 1.26.4 下实测**；归档生产日志
  捆绑在 `../results/`。
- 复现基线协议即论文协议（Table 2: 5 轮；Table 3: 1000 runs；Table 4:
  T=100；Table 5: T=24）。全部跑完约 75 分钟；如只想看代码，不必执行。

**阅读顺序**：环境 → Table 2 → Table 3 → Table 4 → Table 5 → 图 → 总结。
""")

code(r"""
import os
import sys

# Locate the artifact root (this notebook lives in Complex_Count_Sketch/notebooks/).
# Walk upwards from the cwd so the notebook is portable to any checkout.
_here = os.path.abspath(os.getcwd())
ART = None
for _up in range(4):
    _cand = os.path.abspath(os.path.join(_here, *([".."] * _up)))
    if os.path.exists(os.path.join(_cand, "ccs.py")):
        ART = _cand
        break
if ART is None:
    raise RuntimeError("artifact root not found (no ccs.py in cwd or "
                       "its first three parents); run this notebook from "
                       "Complex_Count_Sketch/notebooks/")
sys.path.insert(0, ART)
os.chdir(ART)

import platform
import numpy as np
import ccs
import loaders
print("artifact :", ART)
print("python   :", platform.python_version())
print("numpy    :", np.__version__)
print("data dir :", loaders.DATA)
""")

# ---------------------------------------------------------------- Table 2 - #
md(r"""
## Table 2 — 合成 + 真实数据上的 $F_p$ 估计（`table:lpnorm`）

协议：4 个合成分布（uniform / zipf1.1 / zipf1.5 / mix，$D=10^4$、$N=10^7$）
与三大真实数据集（MovieLens 25M、Airlines 10M、Kasandr merge 列）上的
$p=3,4,5$；每数据集 5 轮，报告 running mean 的相对误差；CCS 用与 C++ 驱动
完全一致的 MurmurHash32 原始行字符串视图。

**取数口径**：论文 Table 2 的相对误差由 C++ 驱动以 6 位有效数字打印的估计值
计算（本包按同一约定复现）；p=4/p=5 合成与全部真实数据集的归档日志已捆绑在
`results/table2/`，可逐位复核（见 `table2_archive_audit.py`）。**AMS 列
不可复现**：C++ 管线用 `std::random_device` 抽位置种子，论文表的 AMS 列
因此不参与逐格对账（CCS 列不受影响）。

预期结果：42 个 CCS 格全部命中论文值（`*`），加粗 pattern 与论文一致。
""")

code(r"""
import inspect
import tables.table2_lpnorm as t2
print(inspect.getsource(t2))
""")

code(r"""
t2.main()
""")

# ---------------------------------------------------------------- Table 3 - #
md(r"""
## Table 3 — T-MCCS vs 原始 CCS（`table:tmccs`）

协议：$\alpha\in\{0,0.5,0.8,1.0,1.1,1.5,2.0\}$ 的截尾 Zipf（$D=10^3$、
$N=10^5$），$w=256$、$d=64$、**1000 runs**，$p=3,4,5$；四个估计量
CCS(plain median) / T-MCCS / Fixed_20 / Rand_20 的绝对相对误差，附
Wilson 区间与符号检验胜率。

**已知偏差（论文侧）**：Fixed20 ($p=3,\alpha=1.0$) 论文值 2.00E-03，
归档日志为 2.454e-03 —— 本包复现归档值，疑为论文转录笔误（该行加粗结论
不受影响）。其余 83/84 格与归档逐位一致。
""")

code(r"""
import inspect
import tables.table3_tmccs as t3
print(inspect.getsource(t3))
""")

code(r"""
t3.main()
""")

# ---------------------------------------------------------------- Table 4 - #
md(r"""
## Table 4 — p-star join 的 T-MCCS 评测（`table:pstar`）

协议：$p=3$、$w=256$、$d=64$、$N=10^7$/列、**T=100 个哈希族**（seed 7919）；
6 个列组合（均匀×均匀×均匀 → Zipf1.5³）。CCS / T-MCCS / G-A 列来自
`rev_pstar.py` 协议（每组合一个 rng，Rand20 的排列抽取保持流对齐）；
Per-edge / Shared-sign 列来自 exp9 协议（种子 8100+t / 8200+t，T=100）。

**移植质量记录**：本包第一版 `table4_pstar.py` 的内联复数乘法曾把虚部写成
`pr*ci + pi*ci`（正确为 `pr*ci + pi*cr`），被归档交叉核对当场抓住；修复后
与原始生产者在 py311 下逐位一致。这也是本对账本存在的意义——**每个数字
都能回到归档日志**。

**唯一显示层差异**：z1.5³ 组合的 CCS 复现值为 3.125e-04，恰落在两位尾数
3.12/3.13 的舍入边界（Python 格式化为 3.12E-04，论文为 3.13E-04）——非复现
偏差，该组合真值 5.91476e+19 与全部诊断量逐位一致。
""")

code(r"""
import inspect
import tables.table4_pstar as t4
print(inspect.getsource(t4))
""")

code(r"""
t4.main()
""")

# ---------------------------------------------------------------- Table 5 - #
md(r"""
## Table 5 — 真实数据上的跨阶回答（`tab:realccs`）

协议：MovieLens 3-star（round-robin 三分片，对齐场景，T=24，seeds 9000+t）
+ 四个真实 2-join 的跨阶回答（p=3,4,5，seeds 4000+p+t，w=1024）与
byte-matched Count Sketch 基线（w=2048，seeds 3000+t）。

**如实披露的取数缺口**：论文的 CCS p=3/4/5 列原始运行未归档，本包试过的
两种种子协议（4000+p 与 4000+100p）都与论文值不符（例：AL Origin-Dest
p=3，论文 2.5E-03，归档协议 2.97E-03）。facsimile 中该列标注为参照而非
复现。3-star 行与 CS 列与归档逐位一致。JoinSketch 列为外部 C++ 基线
（其同预算复现在 `experiments/exp5`，逐位一致）。
""")

code(r"""
import inspect
import tables.table5_realccs as t5
print(inspect.getsource(t5))
""")

code(r"""
t5.main()
""")

# ----------------------------------------------------------------- Figures - #
md(r"""
## 图 1 / 图 2 — width sweep 与 T-MCCS 适用域

两张图在 12 页限额下已从正文移除（PDF 仍在 `papers/figs/`），本包完整复现：

- **fig_sweep**：exp8 协议 part (b)（zipf1.1，$D=10^4$、$N=10^7$、$d=64$、
  10 runs/宽度，seeds 8000+t）—— `--recompute` 从零重算并与归档逐项核对。
- **fig_regime**：exp3 协议（Table 3 的 plain / T-MCCS 列，1000 runs）——
  同样从零重算核对。

下面先展示脚本源码，再全量重算验证（15+36 项，3 位有效数字），最后出图。
""")

code(r"""
import inspect
import tables.figures as figs
print(inspect.getsource(figs))
""")

code(r"""
ok = figs.verify()
assert ok, "recomputed figure data disagrees with the archives"
""")

code(r"""
figs.main()   # plot from the bundled archives (fast path)
from IPython.display import Image, display
for name in ("fig_sweep", "fig_regime"):
    display(Image(filename=os.path.join(ART, "tables", "figs_out",
                                        name + ".png")))
""")

# ----------------------------------------------------------------- Summary - #
md(r"""
## 总结 — 对账清单

| 对象 | 协议 | 结果 |
|---|---|---|
| Table 2 | 5 轮，7 数据集 × p=3,4,5 | 42/42 CCS 格命中论文值；AMS 列不可复现（random_device，论文侧） |
| Table 3 | 1000 runs × 21 配置 | 83/84 逐位一致；Fixed20(p=3,α=1.0) 疑论文转录笔误 |
| Table 4 | T=100 × 6 组合 | 修复共轭 bug 后全列命中（归档 + 论文） |
| Table 5 | T=24 | 3-star + CS 列逐位一致；CCS 列为已知取数缺口（已标注） |
| 图 1/2 | 重算 15+36 项 | 3 位有效数字全部命中归档 |

**为什么可以相信**：每个脚本都是原始生产者的忠实移植（RNG 抽取顺序、种子
基准、协议参数逐一保持），论文值、归档日志、复现值三方对账；归档日志随包
分发（`../results/`）可独立复核。正文其余 16 个实验的复现与对账见
`../experiments/`（`run_all.py --list`）。
""")

nb.cells = cells

out = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "paper_tables.ipynb"))
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    nbf.write(nb, fh)
print("wrote", out, "with", len(cells), "cells")
