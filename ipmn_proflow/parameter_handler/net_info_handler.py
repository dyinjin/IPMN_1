import igraph as ig
import pandas as pd
# import cugraph


def net_info_tic(dataset):
    """
    tic: transactions involved counter
    Augments the dataset with transaction counts for sender and receiver accounts. Optimized with lambda for efficiency.

    Args:
        dataset (pd.DataFrame): The input dataset containing sender and receiver account information.

    Returns:
        pd.DataFrame: The dataset with additional columns for transaction counts:
                      - 'out_same_count': Total transactions (in and out) for the sender account.
                      - 'in_same_count': Total transactions (in and out) for the receiver account.
    """
    # Count the number of outgoing and incoming transactions for all accounts
    outgoing_count = dataset['Sender_account'].value_counts()
    incoming_count = dataset['Receiver_account'].value_counts()

    # Create dictionaries for fast lookups
    outgoing_dict = outgoing_count.to_dict()
    incoming_dict = incoming_count.to_dict()

    # Map transaction counts directly using lambda for optimized performance
    dataset['out_same_count'] = dataset['Sender_account'].map(
        lambda x: outgoing_dict.get(x, 0) + incoming_dict.get(x, 0))
    dataset['in_same_count'] = dataset['Receiver_account'].map(
        lambda x: outgoing_dict.get(x, 0) + incoming_dict.get(x, 0))

    return dataset


def net_info_rtw(dataset):
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

    # Filter transactions where both out_same_count and in_same_count are greater than 1
    dataset_filtered = dataset[(dataset['out_same_count'] > 1) | (dataset['in_same_count'] > 1)].copy()

    # Temporary dictionaries to track the most recent accounts
    sender_last_transaction = {}
    receiver_last_transaction = {}

    # Initialize lists to store the new columns
    sender_recent_with_account = []
    receiver_recent_with_account = []

    # Iterate over the dataset rows to populate recent transaction information (optimized with lists)
    for sender, receiver in zip(dataset_filtered['Sender_account'], dataset_filtered['Receiver_account']):
        # Get the most recent account the sender interacted with
        sender_recent_with_account.append(sender_last_transaction.get(sender, 0))
        # Update the sender's last transaction
        sender_last_transaction[sender] = receiver

        # Get the most recent account the receiver interacted with
        receiver_recent_with_account.append(receiver_last_transaction.get(receiver, 0))
        # Update the receiver's last transaction
        receiver_last_transaction[receiver] = sender

    # Add the new columns to the filtered dataset
    dataset_filtered['Sender_recent_with_account'] = sender_recent_with_account
    dataset_filtered['Receiver_recent_with_account'] = receiver_recent_with_account

    return dataset_filtered


def net_info_rtw_without_filter(dataset):
    """
    Optimized version for filtering transaction data and adding the most recent account information for senders and receivers.

    Args:
        dataset (pd.DataFrame): The input dataset containing transaction information.

    Returns:
        pd.DataFrame: The dataset with additional columns:
                      - 'Sender_recent_with_account': The most recent account the sender has interacted with.
                      - 'Receiver_recent_with_account': The most recent account the receiver has interacted with.
    """

    # Filter transactions where both out_same_count and in_same_count are greater than 1
    dataset_filtered = dataset[(dataset['out_same_count'] > 1) | (dataset['in_same_count'] > 1)].copy()

    # Temporary dictionaries to track the most recent accounts
    sender_last_transaction = {}
    receiver_last_transaction = {}

    # Initialize lists to store the new columns
    sender_recent_with_account = []
    receiver_recent_with_account = []

    # Iterate over the dataset rows to populate recent transaction information (optimized with lists)
    for sender, receiver in zip(dataset_filtered['Sender_account'], dataset_filtered['Receiver_account']):
        # Get the most recent account the sender interacted with
        sender_recent_with_account.append(sender_last_transaction.get(sender, 0))
        # Update the sender's last transaction
        sender_last_transaction[sender] = receiver

        # Get the most recent account the receiver interacted with
        receiver_recent_with_account.append(receiver_last_transaction.get(receiver, 0))
        # Update the receiver's last transaction
        receiver_last_transaction[receiver] = sender

    # Add the new columns to the filtered dataset
    dataset_filtered['Sender_recent_with_account'] = sender_recent_with_account
    dataset_filtered['Receiver_recent_with_account'] = receiver_recent_with_account

    return dataset_filtered


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
    closeness_centrality = G.closeness(normalized=True)
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


# def net_info_3_gpu(dataset):
#     """
#     Creates directed graph based on sender_account and receiver_account, calculates degree, closeness, and betweenness centralities
#     using GPU-accelerated cuGraph, and attaches these features to the transaction dataset.
#
#     Args:
#         dataset (pd.DataFrame): The input dataset containing sender_account and receiver_account columns.
#
#     Returns:
#         pd.DataFrame: The dataset with additional columns for sender and receiver centralities:
#                       - sender_account_degree_centrality
#                       - sender_account_closeness_centrality
#                       - sender_account_betweenness_centrality
#                       - receiver_account_degree_centrality
#                       - receiver_account_closeness_centrality
#                       - receiver_account_betweenness_centrality
#     """
#     # Step 1: Prepare the graph input for cuGraph
#     df_edges = dataset[['Sender_account', 'Receiver_account']].rename(columns={'Sender_account': 'src', 'Receiver_account': 'dst'})
#
#     # Step 2: Create a directed graph using cuGraph
#     g = cugraph.Graph(directed=True)
#     g.from_cudf_edgelist(df_edges, source='src', destination='dst')
#
#     # Step 3: Compute centrality measures using cuGraph
#     # Degree centrality
#     degree_centrality = cugraph.degree_centrality(g).rename(columns={'vertex': 'account', 'degree_centrality': 'degree_centrality'})
#     # Closeness centrality
#     closeness_centrality = cugraph.closeness_centrality(g).rename(columns={'vertex': 'account', 'closeness_centrality': 'closeness_centrality'})
#     # Betweenness centrality
#     betweenness_centrality = cugraph.betweenness_centrality(g).rename(columns={'vertex': 'account', 'betweenness_centrality': 'betweenness_centrality'})
#
#     # Merge centrality measures into a single DataFrame
#     centrality_df = degree_centrality.merge(closeness_centrality, on='account') \
#                                      .merge(betweenness_centrality, on='account')
#
#     # Step 4: Attach centrality measures to the dataset
#     dataset = dataset.merge(centrality_df, left_on='Sender_account', right_on='account', how='left') \
#                      .rename(columns={
#                          'degree_centrality': 'sender_account_degree_centrality',
#                          'closeness_centrality': 'sender_account_closeness_centrality',
#                          'betweenness_centrality': 'sender_account_betweenness_centrality'
#                      }).drop(columns=['account'])
#
#     dataset = dataset.merge(centrality_df, left_on='Receiver_account', right_on='account', how='left') \
#                      .rename(columns={
#                          'degree_centrality': 'receiver_account_degree_centrality',
#                          'closeness_centrality': 'receiver_account_closeness_centrality',
#                          'betweenness_centrality': 'receiver_account_betweenness_centrality'
#                      }).drop(columns=['account'])
#
#     return dataset
