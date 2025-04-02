from ipmn_proflow.imports import *

def net_info_1(dataset):
    """
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


def net_info_2(dataset):
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
