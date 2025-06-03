import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("HI-Small_Trans.csv")

# 随机抽取 30 条数据
df_sample = df.sample(n=1000000, random_state=42)  # 设置 `random_state` 以保证每次抽样结果一致

# 保存为新的 CSV 文件
df_sample.to_csv("sampled_IBM.csv", index=False)
