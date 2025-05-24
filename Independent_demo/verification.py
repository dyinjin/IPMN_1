import pandas as pd

X_train = pd.read_csv("E:/default_download/IPMN_pro/IPMN_1/data/all-tdd_net_info_1-cut_7_train_3_test-X_train.csv")
X_test = pd.read_csv("E:/default_download/IPMN_pro/IPMN_1/data/all-tdd_net_info_1-cut_7_train_3_test-X_test.csv")
y_train = pd.read_csv("E:/default_download/IPMN_pro/IPMN_1/data/all-tdd_net_info_1-cut_7_train_3_test-y_train.csv")
y_test = pd.read_csv("E:/default_download/IPMN_pro/IPMN_1/data/all-tdd_net_info_1-cut_7_train_3_test-y_test.csv")

y_pred = pd.read_csv("E:/default_download/IPMN_pro/IPMN_1/data/all-tdd_net_info_1-cut_7_train_3_test-y_pred.csv")

print(X_test.shape, y_test.shape, y_pred.shape)

# 统计 X_train 中出现的账号
trained_accounts = set(X_train["Sender_account"]).union(set(X_train["Receiver_account"]))

# 提取 X_test 中不含已训练账号的行（即新账号交易）
new_accounts_test = X_test[
    ~X_test["Sender_account"].isin(trained_accounts) |
    ~X_test["Receiver_account"].isin(trained_accounts)
]

print(new_accounts_test)

# 提取对应的 y_test 和 y_pred
y_test_new = y_test.loc[new_accounts_test.index]
y_pred_new = y_pred.loc[new_accounts_test.index]

print(y_test_new.value_counts())
print(y_pred_new.value_counts())

# 计算预测准确率
accuracy_new_accounts = (y_test_new.values == y_pred_new.values).mean()

print(accuracy_new_accounts)
