from ipmn_proflow.imports import *


def net_info_1(dataset):
    """
    Augments the dataset with transaction counts for sender and receiver accounts.

    Args:
        dataset (pd.DataFrame): The input dataset containing sender and receiver account information.

    Returns:
        pd.DataFrame: The dataset with additional columns for transaction counts:
                      - 'out_same_count': Total transactions (in and out) for sender account.
                      - 'in_same_count': Total transactions (in and out) for receiver account.
    """
    # Count the number of outgoing and incoming transactions for all accounts
    outgoing_count = dataset['Sender_account'].value_counts()
    incoming_count = dataset['Receiver_account'].value_counts()

    # Create dictionaries for fast lookups
    outgoing_dict = outgoing_count.to_dict()
    incoming_dict = incoming_count.to_dict()

    # Use the dictionaries to map transaction counts directly
    dataset['out_same_count'] = dataset['Sender_account'].map(lambda x: outgoing_dict.get(x, 0) + incoming_dict.get(x, 0))
    dataset['in_same_count'] = dataset['Receiver_account'].map(lambda x: outgoing_dict.get(x, 0) + incoming_dict.get(x, 0))

    return dataset
