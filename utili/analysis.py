import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
# data = pd.read_csv("quick_test-param_b-X_test_with_y.csv")
data = pd.read_csv("all_d73-param_a-X_train_with_y.csv")

# 选择需要可视化的属性列
columns = [
    "Payment_currency", "Received_currency", "Payment_type",
    "Week", "Day", "Hour", "Minute", "Second",
    "Sender_send_count", "Sender_send_frequency",
    "Receiver_receive_count", "Receiver_receive_frequency",
    "Sender_receive_count", "Sender_receive_frequency",
    "Receiver_send_count", "Receiver_send_frequency", "sender_account_degree_centrality",
    "sender_account_closeness_centrality", "sender_account_betweenness_centrality",
    "receiver_account_degree_centrality", "receiver_account_closeness_centrality",
    "receiver_account_betweenness_centrality"
]

amount_columns = ["Amount", "Sender_send_amount", "Receiver_receive_amount", "Sender_receive_amount", "Receiver_send_amount", ]



# 设置双色柱状图，并对 Y 轴取对数
for col in columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x=col, hue="Is_laundering", multiple="dodge", bins=50, palette=["skyblue", "orange"])
    plt.yscale("log")  # 对 y 轴取对数
    plt.title(f"{col} vs Is_laundering (Log Scale)")
    plt.xlabel(col)
    plt.ylabel("Log(Count)")
    plt.legend(title="Is_laundering", labels=["Yes", "No"])
    plt.show()

# for col in amount_columns:
#     plt.figure(figsize=(8, 5))
#     sns.histplot(data=data, x=np.log10(data[col]), hue="Is_laundering", multiple="dodge", bins=50, palette=["skyblue", "orange"])
#     plt.yscale("log")  # 对 y 轴取对数
#     plt.title(f"log({col}) vs Is_laundering (Log Scale)")
#     plt.xlabel(f"log({col})")
#     plt.ylabel("Log(Count)")
#     plt.legend(title="Is_laundering", labels=["Yes", "No"])
#     plt.show()