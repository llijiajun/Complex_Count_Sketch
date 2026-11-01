# Complex_Count_Sketch:代码与数据发布

论文 *Complex Count Sketch: Mergeable Frequency-Moment and p-Star Join Size
Estimation*(PVLDB)的可复现实验包:自带全部所需数据,一个共享核心库,
20 个实验脚本与归档结果。

## 目录

```
Complex_Count_Sketch/
├── ccs.py            # 核心库:草图构建 / 各估计器 / T-MCCS / 精确验证工具
├── loaders.py        # 数据加载(gzip 透明解压;Kasandr 预提取 npz)
├── data/             # 全部数据(约 330 MB,含来源说明,见 data/README.md)
│   ├── ML/ratings/*.gz, ML/tags/*.gz     # MovieLens 列文件
│   ├── airlines/*.gz                     # Airlines 10 列
│   ├── lp/*.gz                           # 合成列(uniform/zipf1.1/zipf1.5 ×3)
│   └── kasandr_essential.npz             # Kasandr users/cats/times 预提取
├── experiments/      # 20 个实验脚本(与论文位置一一对应,见下表)
└── results/          # 归档输出;legacy/ 为初版实验套件的原始输出
```

## 快速查阅

外层笔记本 `ccs_notebook.ipynb`(随发布一起提供)是全部核心实验的
可执行走读版:草图机制、方差核验、实数乘子、F_p 扫描、T-MCCS、
p-star 基线、MovieLens 一个部署三个任务、错位实验,已带运行输出。

## 运行

```bash
cd experiments
python3 run_all.py exp01_fp_skew      # 单个实验
python3 run_all.py                    # 全部(数小时)
```

依赖:numpy(scipy 仅 exp05/exp07 需要)。

## 实验与论文对照

| 脚本 | 论文位置 | 内容 |
|---|---|---|
| exp01_fp_skew | §5.2 表 2 CCS 列 | 合成数据 F_p + 宽度扫描 + CI |
| exp02_tmccs_table | §5.4 表 4 | median / T-MCCS / Fixed20 |
| exp03_aggregators | §5.4 | median vs mean vs MoM vs T-MCCS |
| exp04_pstar_baselines | §5.5 表 5 | p-star:CCS / per-edge / shared-sign |
| exp05_misalignment | §5.5 | 固定边际、重频位移 L 网格 + 精确 Ψ |
| exp06_pattern_check | §4.4 式(13) | 共址模式精确枚举验证 |
| exp07_real_solutions | §4.2 | 实数乘子:精确解 + 精确期望 + 方差比 |
| exp08_real_ml | §5.7 | MovieLens 真实连接(3-star) |
| exp09_kasandr | §5.7 | Kasandr 错位 3-star + 基线 |
| exp10_merge_delete | §5.9 | 16 站点合并 + 删除 |
| exp12_ganguly_attribution | §5.2 | Ganguly 差距归因 2×2(分配×聚合) |
| exp13_joint_workload | §5.8 | 一套草图答 F3/2-join/3-star |
| exp14_equal_budget | §5.2 | CCS 缩至 AMS 预算 |
| exp15_five_star | §5.7 | 五表 star(order-5) |
| exp16_bridge_demo | §5.7 | 桥接连接(卷积 + 素数步长投影) |
| exp19_tmccs_on_peredge | §5.5 信号边界 | 能量代理与行误差负相关(−0.22):虚部信号不可移植到实数方案 |
| exp20_fp_sizing | §5.3 表 3 | F_p sizing:闭式系数、prior/fixed/ideal/pilot、150 部署/行 |
| exp21_star_sizing | §5.6 表 6 | 3-star sizing:A、B 线性时间展开 + floor 不可行证书 |
| exp22_single_state_real3 | §5.8 表 7 | 单状态 real-3 与 CCS 的七任务联合达标比较(E_max 协议) |

脚本编号存在空缺(exp11、exp17、exp18),对应初版套件中已从论文删除或
并入其他实验的条目;`results/legacy/` 保留了初版套件的原始输出,
文件名沿用当时的编号。

## 数据来源与再生成

- MovieLens:Harper & Konstan (2015),按列切分,打包为 gzip。
- Airlines:OpenML 42728,按列切分,gzip。
- Kasandr:Sidana et al. (2017) train_de.csv(2.9 GB);发布包仅保留
  userid/category/timestamp 三列的整数编码(`kasandr_essential.npz`,
  53 MB),与原 CSV 逐行对应。
- 合成列(lp/*.dat*):10^7 整数/列,原生成脚本未固定种子,故发布包
  直接打包原始文件(gzip);`data/README.md` 有逐文件基数与峰值频数。

## results/ 的构成与论文数值的对应

- `results/exp20_*.out`、`exp21_*.out`、`exp22_*.out` 由发布包内脚本
  在论文协议下直接生成,可复跑复验。
- `results/legacy/` 保存初版实验套件产生、且论文正文引用的原始输出
  (Ganguly 四格、错位网格的 J/Ψ、合并删除、真实连接等);旧套件编号
  与发布脚本编号不同,对应关系为 exp1→exp08、exp3→exp02、exp4→exp14、
  exp11→exp06、exp13→exp05、exp14→exp10、exp16→exp07、exp17→exp12、
  exp18→exp13。
- AMS / PS-Lp / LpS / JoinSketch 列来自外部基线的原始实现,不在本包内
  重跑;其数字与运行配置见 `results/legacy/` 与论文 §5.1。
