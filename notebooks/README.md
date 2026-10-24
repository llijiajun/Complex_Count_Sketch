# notebooks/ — 论文表格对账本

`paper_tables.ipynb` 是**给投稿人逐格核对论文数字**用的对账本：论文四张表格
（Table 2–5）与两张图（fig_sweep / fig_regime）的实验代码集中在一处，先展示
脚本源码、再原样运行，运行输出即论文值 facsimile —— 每格并列"复现值 / 发表值"，
命中标注 `*`，偏离标注 `!`。

## 怎么用

```bash
cd Complex_Count_Sketch/notebooks
jupyter lab paper_tables.ipynb      # 或 jupyter nbclassic / VS Code
```

打开即可阅读：**交付版已带全部执行输出**（2026-10-23 在 py311 / numpy 1.26.4
下实测），不必自己再跑一遍。若要重跑，用 *Run All*：

- Table 2 ~1 min、Table 3 ~4 min、**Table 4 ~35 min（长尾）**、Table 5 ~2 min、
  图（重算核对）~2 min —— 合计约 45–75 分钟，取决于机器。
- 若只想看代码与已存输出，**不要 Run All**。

## 结构（19 cells）

| cell | 内容 |
|---|---|
| 1 / 3 / 6 / 9 / 12 / 15 | `inspect.getsource(...)` 展示脚本源码（Table 2/3/4/5、figures） |
| 4 / 7 / 10 / 13 | 对应脚本的 `main()` 实跑 + facsimile |
| 16 | `figs.verify()` —— 从零重算图数据并与归档逐项核对（15+36 项，3 位有效数字） |
| 17 | `figs.main()` 出图并内嵌显示 |
| 0 / 2 / 5 / 8 / 11 / 14 / 18 | 各表协议、取数口径、已知偏差的如实说明 + 总结清单 |

## 与脚本不会漂移

notebook 通过 `inspect.getsource` 展示并调用 `../tables/` 下的脚本本体，
**不做任何拷贝**；脚本改了，notebook 里看到的就是新的。

## 重新生成 / 重新执行

```bash
python notebooks/build_notebook.py     # 重新生成未执行的 ipynb（会覆盖）
```

执行用 nbclient 逐 cell 落盘（见 `build_notebook.py` 头注释里提到的做法）；
Table 4 单 cell 需把超时放到 ≥2 小时，否则会被判超时并丢掉全部输出。

## 已知差异（notebook 里也各自标注）

1. **Table 3** Fixed20 ($p=3,\alpha=1.0$)：论文 2.00E-03，归档与本包复现
   2.454e-03 —— 疑论文转录笔误，该行加粗结论不受影响；其余 83/84 逐位一致。
2. **Table 4** z1.5³ 的 CCS 值 3.125e-04 恰落在 3.12/3.13 舍入边界
   （Python 格式化 3.12E-04 vs 论文 3.13E-04），非复现偏差。
3. **Table 5** 论文的 CCS p=3/4/5 列原始运行未归档，两种候选种子协议都与
   论文值不符 —— facsimile 中该列标为参照（`!`），不是复现结果。
4. **Table 2** 的 AMS 列不可复现（C++ 侧用 `std::random_device`），不参与
   逐格对账；CCS 列 42/42 命中。
