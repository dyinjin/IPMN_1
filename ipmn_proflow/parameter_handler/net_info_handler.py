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
    # Create a copy of the dataset for processing
    df = dataset

    # Count the number of outgoing transactions for each sender account
    outgoing_count = df['Sender_account'].value_counts()

    # Count the number of incoming transactions for each receiver account
    incoming_count = df['Receiver_account'].value_counts()

    def calculate_counts(row):
        """
        Calculate total transaction counts for sender and receiver accounts.

        Args:
            row (pd.Series): A single row from the dataset.

        Returns:
            pd.Series: A Series containing:
                       - 'out_same_count': Total transactions for sender account.
                       - 'in_same_count': Total transactions for receiver account.
        """
        # Total transactions for the sender account (outgoing + incoming)
        total_sender_count = (outgoing_count.get(row['Sender_account'], 0) +
                              incoming_count.get(row['Sender_account'], 0))

        # Total transactions for the receiver account (outgoing + incoming)
        total_receiver_count = (outgoing_count.get(row['Receiver_account'], 0) +
                                incoming_count.get(row['Receiver_account'], 0))

        return pd.Series({
            'out_same_count': total_sender_count,
            'in_same_count': total_receiver_count
        })

    # Apply the `calculate_counts` function across all rows to generate new columns
    df[['out_same_count', 'in_same_count']] = df.apply(calculate_counts, axis=1)

    # Return the augmented dataset
    return df
