import igraph as ig
import numpy as np
import pandas as pd
from collections import defaultdict, deque


def window_before(dataset, window_days):
    """
    按时间窗口划分数据，并分别调用 net_info_rti() 进行处理
    :param dataset: 原始交易数据集
    :param window_days: 时间窗口天数
    :return: 处理后的数据集
    """

    start_date = dataset["Date"].min()
    end_date = dataset["Date"].max()
    total_days = (end_date - start_date).days + 1

    processed_dfs = []  # 存储每个时间窗口处理后的数据

    # 逐步处理时间窗口
    for i in range(0, total_days, window_days):
        window_start = start_date + pd.Timedelta(days=i)
        window_end = min(start_date + pd.Timedelta(days=i + window_days - 1), end_date)

        # 过滤出当前时间窗口的数据
        window_df = dataset[(dataset["Date"] >= window_start) & (dataset["Date"] <= window_end)].copy()
        print(f"window df length:{len(window_df)}")

        # 仅在窗口数据非空时处理
        if not window_df.empty:
            processed_window_df = net_info_rti(window_df)
            processed_dfs.append(processed_window_df)

    # 合并所有处理后的数据
    final_dataset = pd.concat(processed_dfs, ignore_index=True)
    return final_dataset


def window_before_graph(dataset, window_days):
    start_date = dataset["Date"].min()
    end_date = dataset["Date"].max()
    total_days = (end_date - start_date).days + 1

    processed_dfs = []  # 存储每个时间窗口处理后的数据

    # 逐步处理时间窗口
    for i in range(0, total_days, window_days):
        window_start = start_date + pd.Timedelta(days=i)
        window_end = min(start_date + pd.Timedelta(days=i + window_days - 1), end_date)

        # 过滤出当前时间窗口的数据
        window_df = dataset[(dataset["Date"] >= window_start) & (dataset["Date"] <= window_end)].copy()
        print(f"window df length:{len(window_df)}")

        # 仅在窗口数据非空时处理
        if not window_df.empty:
            processed_window_df = net_info_3centrality(window_df)
            processed_dfs.append(processed_window_df)
            del processed_window_df

    # 合并所有处理后的数据
    final_dataset = pd.concat(processed_dfs, ignore_index=True)
    del processed_dfs
    return final_dataset


def window_before_inte(dataset, window_days):
    start_date = dataset["Date"].min()
    end_date = dataset["Date"].max()
    total_days = (end_date - start_date).days + 1

    processed_dfs = []  # 存储每个时间窗口处理后的数据

    # 逐步处理时间窗口
    for i in range(0, total_days, window_days):
        window_start = start_date + pd.Timedelta(days=i)
        window_end = min(start_date + pd.Timedelta(days=i + window_days - 1), end_date)

        # 过滤出当前时间窗口的数据
        window_df = dataset[(dataset["Date"] >= window_start) & (dataset["Date"] <= window_end)].copy()
        print(f"window df length:{len(window_df)}")

        # 仅在窗口数据非空时处理
        if not window_df.empty:
            processed_window_df = net_info_3centrality(window_df)
            processed_window_df = net_info_rti(processed_window_df)
            processed_dfs.append(processed_window_df)
            del processed_window_df

    # 合并所有处理后的数据
    final_dataset = pd.concat(processed_dfs, ignore_index=True)
    del processed_dfs
    return final_dataset


def window_slider(dataset, window_days, step):
    """
    滑动窗口计算交易统计信息，并确保后续窗口计算结果正确覆盖已有数据，而不产生新行
    """
    dataset[
        ['Sender_send_amount', 'Sender_send_count', 'Sender_send_frequency', 'Receiver_receive_amount',
         'Receiver_receive_count', 'Receiver_receive_frequency', 'Sender_receive_amount', 'Sender_receive_count',
         'Sender_receive_frequency', 'Receiver_send_amount', 'Receiver_send_count', 'Receiver_send_frequency']
    ] = 0.0
    start_date = dataset["Date"].min()
    end_date = dataset["Date"].max()
    total_days = (end_date - start_date).days + 1

    # 滑动窗口计算
    for i in range(total_days - window_days, -1, -step):  # 控制窗口滑动范围
        window_start = start_date + pd.Timedelta(days=i)
        window_end = window_start + pd.Timedelta(days=window_days - 1)

        window_df = dataset[(dataset["Date"] >= window_start) & (dataset["Date"] <= window_end)].copy()
        print(f"window df length:{len(window_df)}")

        # 仅在窗口数据非空时处理
        if not window_df.empty:
            processed_window_df = net_info_rti(window_df[['Date', 'Sender_account', 'Receiver_account', 'Amount']])
            processed_window_df = processed_window_df.set_index(window_df.index)

            # print(dataset.shape)
            # print(processed_window_df.shape)
            # dataset = dataset.combine_first(processed_window_df)
            dataset.update(processed_window_df)

            del processed_window_df
        del window_df

    return dataset


def window_slider_graph(dataset, window_days, step):
    dataset[
        ['sender_account_degree_centrality', 'sender_account_closeness_centrality',
         'sender_account_betweenness_centrality', 'receiver_account_degree_centrality',
         'receiver_account_closeness_centrality', 'receiver_account_betweenness_centrality']
    ] = 0.0
    start_date = dataset["Date"].min()
    end_date = dataset["Date"].max()
    total_days = (end_date - start_date).days + 1

    # 滑动窗口计算
    for i in range(total_days - window_days, -1, -step):  # 控制窗口滑动范围
        window_start = start_date + pd.Timedelta(days=i)
        window_end = window_start + pd.Timedelta(days=window_days - 1)

        window_df = dataset[(dataset["Date"] >= window_start) & (dataset["Date"] <= window_end)].copy()
        print(f"window df length:{len(window_df)}")

        # 仅在窗口数据非空时处理
        if not window_df.empty:
            processed_window_df = net_info_3centrality(window_df[['Date', 'Sender_account', 'Receiver_account', 'Amount']])
            processed_window_df = processed_window_df.set_index(window_df.index)
            dataset.update(processed_window_df)
            del processed_window_df
        del window_df

    return dataset


def window_slider_inte(dataset, window_days, step):
    dataset[
        ['Sender_send_amount', 'Sender_send_count', 'Sender_send_frequency', 'Receiver_receive_amount',
         'Receiver_receive_count', 'Receiver_receive_frequency', 'Sender_receive_amount', 'Sender_receive_count',
         'Sender_receive_frequency', 'Receiver_send_amount', 'Receiver_send_count', 'Receiver_send_frequency',
         'sender_account_degree_centrality', 'sender_account_closeness_centrality',
         'sender_account_betweenness_centrality', 'receiver_account_degree_centrality',
         'receiver_account_closeness_centrality', 'receiver_account_betweenness_centrality']
    ] = 0.0
    start_date = dataset["Date"].min()
    end_date = dataset["Date"].max()
    total_days = (end_date - start_date).days + 1

    # 滑动窗口计算
    for i in range(total_days - window_days, -1, -step):  # 控制窗口滑动范围
        window_start = start_date + pd.Timedelta(days=i)
        window_end = window_start + pd.Timedelta(days=window_days - 1)

        window_df = dataset[(dataset["Date"] >= window_start) & (dataset["Date"] <= window_end)].copy()
        print(f"window df length:{len(window_df)}")

        # 仅在窗口数据非空时处理
        if not window_df.empty:
            processed_window_df = net_info_rti(window_df[['Date', 'Sender_account', 'Receiver_account', 'Amount']])
            processed_window_df = processed_window_df.set_index(window_df.index)
            dataset.update(processed_window_df)
            del processed_window_df

            processed_window_df = net_info_3centrality(window_df[['Date', 'Sender_account', 'Receiver_account', 'Amount']])
            processed_window_df = processed_window_df.set_index(window_df.index)
            dataset.update(processed_window_df)
            del processed_window_df
        del window_df

    return dataset


def net_info_rti(dataset):
    """
    tri: recently trans info
    计算 Sender、Receiver 的交易统计信息，并考虑交叉交易情况。
    """

    # 计算数据集的总时长（天数）
    total_days = (dataset["Date"].max() - dataset["Date"].min()).days + 1  # 计算数据集的总天数

    # 统计 Sender 发送交易信息
    sender_stats = dataset.groupby("Sender_account").agg(
        Sender_send_amount=("Amount", "sum"),
        Sender_send_count=("Amount", "count")
    ).reset_index()
    sender_stats["Sender_send_frequency"] = sender_stats["Sender_send_count"] / total_days

    # 统计 Receiver 接收交易信息
    receiver_stats = dataset.groupby("Receiver_account").agg(
        Receiver_receive_amount=("Amount", "sum"),
        Receiver_receive_count=("Amount", "count")
    ).reset_index()
    receiver_stats["Receiver_receive_frequency"] = receiver_stats["Receiver_receive_count"] / total_days

    # 统计 Sender 作为 Receiver 时的交易信息（Sender_receive_）
    sender_receive_stats = dataset.groupby("Receiver_account").agg(
        Sender_receive_amount=("Amount", "sum"),
        Sender_receive_count=("Amount", "count")
    ).reset_index()
    sender_receive_stats.rename(columns={"Receiver_account": "Sender_account"}, inplace=True)
    sender_receive_stats["Sender_receive_frequency"] = sender_receive_stats["Sender_receive_count"] / total_days

    # 统计 Receiver 作为 Sender 时的交易信息（Receiver_send_）
    receiver_send_stats = dataset.groupby("Sender_account").agg(
        Receiver_send_amount=("Amount", "sum"),
        Receiver_send_count=("Amount", "count")
    ).reset_index()
    receiver_send_stats.rename(columns={"Sender_account": "Receiver_account"}, inplace=True)
    receiver_send_stats["Receiver_send_frequency"] = receiver_send_stats["Receiver_send_count"] / total_days

    # 合并计算结果到原始数据集
    dataset = dataset.merge(sender_stats, on="Sender_account", how="left")
    dataset = dataset.merge(receiver_stats, on="Receiver_account", how="left")
    dataset = dataset.merge(sender_receive_stats, on="Sender_account", how="left")
    dataset = dataset.merge(receiver_send_stats, on="Receiver_account", how="left")

    return dataset


def net_info_tic(dataset):
    """
    tic: transactions involved counter
    Augments the dataset with transaction counts for sender and receiver accounts. Optimized with lambda for efficiency.

    Args:
        dataset (pd.DataFrame): The input dataset containing sender and receiver account information.

    Returns:
        pd.DataFrame: The dataset with additional columns for transaction counts:
                      - 'Sender_count': Total transactions (in and out) for the sender account.
                      - 'Receiver_count': Total transactions (in and out) for the receiver account.
    """
    # Count the number of outgoing and incoming transactions for all accounts
    outgoing_count = dataset['Sender_account'].value_counts()
    incoming_count = dataset['Receiver_account'].value_counts()

    # Create dictionaries for fast lookups
    outgoing_dict = outgoing_count.to_dict()
    incoming_dict = incoming_count.to_dict()

    # Map transaction counts directly using lambda for optimized performance
    dataset['Sender_count'] = dataset['Sender_account'].map(
        lambda x: outgoing_dict.get(x, 0) + incoming_dict.get(x, 0))
    dataset['Receiver_count'] = dataset['Receiver_account'].map(
        lambda x: outgoing_dict.get(x, 0) + incoming_dict.get(x, 0))

    return dataset


def net_info_3centrality(dataset):
    """
    Creates directed graph based on sender_account and receiver_account, calculates degree, closeness, and betweenness centralities,
    and attaches these features to the transaction dataset.

    Args:
        dataset (pd.DataFrame): The input dataset containing sender_account and receiver_account columns.

    Returns:
        pd.DataFrame: The dataset with additional columns for sender and receiver centralities:
                      - sender_account_degree_centrality
                      - sender_account_closeness_centrality
                      - sender_account_betweenness_centrality
                      - receiver_account_degree_centrality
                      - receiver_account_closeness_centrality
                      - receiver_account_betweenness_centrality
    """
    # Step 1: Create a directed graph
    # Extract edges from the dataset
    edges = list(zip(dataset['Sender_account'], dataset['Receiver_account']))

    # Create an igraph directed graph
    G = ig.Graph.TupleList(edges, directed=True)

    # Step 2: Compute centrality measures
    # Degree centrality
    degree_centrality = G.degree(mode="all")  # Includes both in-degree and out-degree
    # Closeness centrality
    closeness_centrality = G.closeness(normalized=False)
    # Betweenness centrality
    betweenness_centrality = G.betweenness()

    # Step 3: Map centrality measures to dataset rows
    # Map the centrality values to the corresponding accounts
    account_list = G.vs["name"]  # Get the list of vertex names (accounts)
    centrality_df = pd.DataFrame({
        "account": account_list,
        "degree_centrality": degree_centrality,
        "closeness_centrality": closeness_centrality,
        "betweenness_centrality": betweenness_centrality
    })

    # Add sender and receiver centralities to the dataset
    dataset = dataset.merge(centrality_df, left_on="Sender_account", right_on="account", how="left") \
                     .rename(columns={
                         "degree_centrality": "sender_account_degree_centrality",
                         "closeness_centrality": "sender_account_closeness_centrality",
                         "betweenness_centrality": "sender_account_betweenness_centrality"
                     }).drop(columns=["account"])

    dataset = dataset.merge(centrality_df, left_on="Receiver_account", right_on="account", how="left") \
                     .rename(columns={
                         "degree_centrality": "receiver_account_degree_centrality",
                         "closeness_centrality": "receiver_account_closeness_centrality",
                         "betweenness_centrality": "receiver_account_betweenness_centrality"
                     }).drop(columns=["account"])

    return dataset


def strict_before_(dataset, window_days):
    sender_history = defaultdict(lambda: defaultdict(int))
    receiver_history = defaultdict(lambda: defaultdict(int))

    sender_queue = deque()
    receiver_queue = deque()

    sender_transaction_count = []
    sender_total_amount = []
    sender_transaction_frequency = []

    receiver_transaction_count = []
    receiver_total_amount = []
    receiver_transaction_frequency = []

    window_duration = pd.Timedelta(days=window_days)

    for _, row in dataset.iterrows():
        sender = row["Sender_account"]
        receiver = row["Receiver_account"]
        date = row["Date"]
        amount = row["Amount"]

        while sender_queue and sender_queue[0][0] < date - window_duration:
            old_date, old_sender = sender_queue.popleft()
            sender_history[old_date].pop(old_sender, None)
            sender_history[old_date].pop(f"{old_sender}_count", None)
            if not sender_history[old_date]:
                del sender_history[old_date]

        while receiver_queue and receiver_queue[0][0] < date - window_duration:
            old_date, old_receiver = receiver_queue.popleft()
            receiver_history[old_date].pop(old_receiver, None)
            receiver_history[old_date].pop(f"{old_receiver}_count", None)
            if not receiver_history[old_date]:
                del receiver_history[old_date]

        sender_history[date][sender] += amount
        sender_history[date][f"{sender}_count"] += 1
        sender_queue.append((date, sender))

        receiver_history[date][receiver] += amount
        receiver_history[date][f"{receiver}_count"] += 1
        receiver_queue.append((date, receiver))

        sender_total = sum(sender_history[d].get(sender, 0) for d in sender_history)
        sender_count = sum(sender_history[d].get(f"{sender}_count", 0) for d in sender_history)
        sender_freq = sender_count / window_days

        receiver_total = sum(receiver_history[d].get(receiver, 0) for d in receiver_history)
        receiver_count = sum(receiver_history[d].get(f"{receiver}_count", 0) for d in receiver_history)
        receiver_freq = receiver_count / window_days

        sender_total_amount.append(sender_total)
        sender_transaction_count.append(sender_count)
        sender_transaction_frequency.append(sender_freq)

        receiver_total_amount.append(receiver_total)
        receiver_transaction_count.append(receiver_count)
        receiver_transaction_frequency.append(receiver_freq)

    dataset["Sender_total_amount"] = sender_total_amount
    dataset["Sender_transaction_count"] = sender_transaction_count
    dataset["Sender_transaction_frequency"] = sender_transaction_frequency

    dataset["Receiver_total_amount"] = receiver_total_amount
    dataset["Receiver_transaction_count"] = receiver_transaction_count
    dataset["Receiver_transaction_frequency"] = receiver_transaction_frequency

    return dataset


def strict_before_graph_(dataset, window_days):
    sender_history = defaultdict(lambda: defaultdict(int))
    receiver_history = defaultdict(lambda: defaultdict(int))

    sender_queue = deque()
    receiver_queue = deque()

    sender_transaction_count = []
    sender_total_amount = []
    sender_transaction_frequency = []

    receiver_transaction_count = []
    receiver_total_amount = []
    receiver_transaction_frequency = []

    sender_centralities = []
    receiver_centralities = []

    window_duration = pd.Timedelta(days=window_days)

    for _, row in dataset.iterrows():
        sender = row["Sender_account"]
        receiver = row["Receiver_account"]
        date = row["Date"]
        amount = row["Amount"]

        # 清理过期数据
        while sender_queue and sender_queue[0][0] < date - window_duration:
            old_date, old_sender = sender_queue.popleft()
            sender_history[old_date].pop(old_sender, None)
            sender_history[old_date].pop(f"{old_sender}_count", None)
            if not sender_history[old_date]:
                del sender_history[old_date]

        while receiver_queue and receiver_queue[0][0] < date - window_duration:
            old_date, old_receiver = receiver_queue.popleft()
            receiver_history[old_date].pop(old_receiver, None)
            receiver_history[old_date].pop(f"{old_receiver}_count", None)
            if not receiver_history[old_date]:
                del receiver_history[old_date]

        # 更新历史记录
        sender_history[date][sender] += amount
        sender_history[date][f"{sender}_count"] += 1
        sender_queue.append((date, sender))

        receiver_history[date][receiver] += amount
        receiver_history[date][f"{receiver}_count"] += 1
        receiver_queue.append((date, receiver))

        # 计算过去 window_days 内的统计信息
        sender_total = sum(sender_history[d].get(sender, 0) for d in sender_history)
        sender_count = sum(sender_history[d].get(f"{sender}_count", 0) for d in sender_history)
        sender_freq = sender_count / window_days

        receiver_total = sum(receiver_history[d].get(receiver, 0) for d in receiver_history)
        receiver_count = sum(receiver_history[d].get(f"{receiver}_count", 0) for d in receiver_history)
        receiver_freq = receiver_count / window_days

        sender_total_amount.append(sender_total)
        sender_transaction_count.append(sender_count)
        sender_transaction_frequency.append(sender_freq)

        receiver_total_amount.append(receiver_total)
        receiver_transaction_count.append(receiver_count)
        receiver_transaction_frequency.append(receiver_freq)

        # 构建窗口期内的交易图并计算中心性
        window_edges = [
            (s, r) for d in sender_history
            for s, r in zip(
                [key for key in sender_history[d].keys() if isinstance(key, int)],
                [key for key in receiver_history[d].keys() if isinstance(key, int)]
            )
        ]

        print(len(window_edges))
        # too slow!

        if window_edges:
            G = ig.Graph.TupleList(window_edges, directed=True)

            account_list = G.vs["name"]
            degree_centrality = G.degree(mode="all")
            closeness_centrality = G.closeness(normalized=True)
            betweenness_centrality = G.betweenness(directed=True)

            centrality_df = pd.DataFrame({
                "account": account_list,
                "degree_centrality": degree_centrality,
                "closeness_centrality": closeness_centrality,
                "betweenness_centrality": betweenness_centrality
            })

            sender_centrality = centrality_df.loc[
                centrality_df["account"] == sender, ["degree_centrality", "closeness_centrality",
                                                     "betweenness_centrality"]].values.flatten() if sender in account_list else [
                np.nan, np.nan, np.nan]
            receiver_centrality = centrality_df.loc[
                centrality_df["account"] == receiver, ["degree_centrality", "closeness_centrality",
                                                       "betweenness_centrality"]].values.flatten() if receiver in account_list else [
                np.nan, np.nan, np.nan]

            del G

        else:
            sender_centrality = [np.nan, np.nan, np.nan]
            receiver_centrality = [np.nan, np.nan, np.nan]

        sender_centralities.append(sender_centrality)
        receiver_centralities.append(receiver_centrality)

    dataset["Sender_total_amount"] = sender_total_amount
    dataset["Sender_transaction_count"] = sender_transaction_count
    dataset["Sender_transaction_frequency"] = sender_transaction_frequency

    dataset["Receiver_total_amount"] = receiver_total_amount
    dataset["Receiver_transaction_count"] = receiver_transaction_count
    dataset["Receiver_transaction_frequency"] = receiver_transaction_frequency

    dataset[
        ["Sender_degree_centrality", "Sender_closeness_centrality", "Sender_betweenness_centrality"]] = pd.DataFrame(
        sender_centralities, index=dataset.index)
    dataset[["Receiver_degree_centrality", "Receiver_closeness_centrality",
             "Receiver_betweenness_centrality"]] = pd.DataFrame(receiver_centralities, index=dataset.index)

    return dataset


def normalize(lst):
    min_val, max_val = min(lst), max(lst)
    val_gap = max_val - min_val
    return [(x - min_val) / val_gap for x in lst]


def net_info_3centrality_normalized_(dataset):
    """
    Creates directed graph based on sender_account and receiver_account, calculates degree, closeness, and betweenness centralities,
    and attaches these features to the transaction dataset.

    Args:
        dataset (pd.DataFrame): The input dataset containing sender_account and receiver_account columns.

    Returns:
        pd.DataFrame: The dataset with additional columns for sender and receiver centralities:
                      - sender_account_degree_centrality
                      - sender_account_closeness_centrality
                      - sender_account_betweenness_centrality
                      - receiver_account_degree_centrality
                      - receiver_account_closeness_centrality
                      - receiver_account_betweenness_centrality
    """
    # Step 1: Create a directed graph
    # Extract edges from the dataset
    edges = list(zip(dataset['Sender_account'], dataset['Receiver_account']))

    # Create an igraph directed graph
    G = ig.Graph.TupleList(edges, directed=True)

    # Step 2: Compute centrality measures
    # Degree centrality
    degree_centrality = G.degree(mode="all")  # Includes both in-degree and out-degree
    degree_centrality = normalize(degree_centrality)
    # Closeness centrality
    closeness_centrality = G.closeness(normalized=True)
    # Betweenness centrality
    betweenness_centrality = G.betweenness()
    betweenness_centrality = normalize(betweenness_centrality)

    # Step 3: Map centrality measures to dataset rows
    # Map the centrality values to the corresponding accounts
    account_list = G.vs["name"]  # Get the list of vertex names (accounts)
    centrality_df = pd.DataFrame({
        "account": account_list,
        "degree_centrality": degree_centrality,
        "closeness_centrality": closeness_centrality,
        "betweenness_centrality": betweenness_centrality
    })

    # Add sender and receiver centralities to the dataset
    dataset = dataset.merge(centrality_df, left_on="Sender_account", right_on="account", how="left") \
                     .rename(columns={
                         "degree_centrality": "sender_account_degree_centrality",
                         "closeness_centrality": "sender_account_closeness_centrality",
                         "betweenness_centrality": "sender_account_betweenness_centrality"
                     }).drop(columns=["account"])

    dataset = dataset.merge(centrality_df, left_on="Receiver_account", right_on="account", how="left") \
                     .rename(columns={
                         "degree_centrality": "receiver_account_degree_centrality",
                         "closeness_centrality": "receiver_account_closeness_centrality",
                         "betweenness_centrality": "receiver_account_betweenness_centrality"
                     }).drop(columns=["account"])

    return dataset


def net_info_rtw_(dataset):
    # not useful
    """
    rtw: recently trans with
    Optimized version for filtering transaction data and adding the most recent account information for senders and receivers.

    Args:
        dataset (pd.DataFrame): The input dataset containing transaction information.

    Returns:
        pd.DataFrame: The dataset with additional columns:
                      - 'Sender_recent_with_account': The most recent account the sender has interacted with.
                      - 'Receiver_recent_with_account': The most recent account the receiver has interacted with.
    """
    # Temporary dictionaries to track the most recent accounts
    sender_last_transaction = {}
    receiver_last_transaction = {}

    # Initialize lists to store the new columns
    sender_recent_with_account = []
    receiver_recent_with_account = []

    # Iterate over the dataset rows to populate recent transaction information (optimized with lists)
    for sender, receiver in zip(dataset['Sender_account'], dataset['Receiver_account']):
        # Get the most recent account the sender interacted with
        sender_recent_with_account.append(sender_last_transaction.get(sender, 0))
        # Update the sender's last transaction
        sender_last_transaction[sender] = receiver

        # Get the most recent account the receiver interacted with
        receiver_recent_with_account.append(receiver_last_transaction.get(receiver, 0))
        # Update the receiver's last transaction
        receiver_last_transaction[receiver] = sender

    # Add the new columns to the filtered dataset
    dataset['Sender_recent_with_account'] = sender_recent_with_account
    dataset['Receiver_recent_with_account'] = receiver_recent_with_account

    return dataset