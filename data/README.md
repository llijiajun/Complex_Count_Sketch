# data/ 说明

| 文件 | 来源 | 规模 | 说明 |
|---|---|---|---|
| ML/ratings/userId.gz, movieId.gz | MovieLens (Harper & Konstan 2015) ratings 表按列切分 | 各 25.0M 行 | 3-star 分片、2-join、F3 实验的输入 |
| ML/tags/userId.gz, movieId.gz | 同上 tags 表 | 各 1.1M 行 | 2-join 的另一侧 |
| airlines/*.gz (10 列) | OpenML 42728 (Airlines Delay) | 各 ~10.0M 行 | 列排名实验 |
| lp/uniform.dat{,2,3}.gz | 均匀 [0,10^4] | 各 10^7 整数 | 合成 p-star 列(三列独立再生成) |
| lp/zipf1.1.dat{,2,3}.gz | Zipf(1.1) 截断到键域 | 各 10^7 整数;实测基数 ~2.8M | 同上 |
| lp/zipf1.5.dat{,2,3}.gz | Zipf(1.5) | 各 10^7 整数;实测基数 ~6.5e4 | 同上 |
| kasandr_essential.npz | KASANDR (Sidana et al. 2017) train_de.csv | 15,844,717 行 | users(int32)/cats(int32)/times(int32,YYYYMMDD)/n_users;与原 CSV 逐行对应 |

原生成脚本(Lp/gen.py 等)未固定随机种子,因此发布包直接打包原始
数据文件而非再生成。解压后的中间文件不需要落盘——`loaders.py` 对
`.gz` 透明处理。
