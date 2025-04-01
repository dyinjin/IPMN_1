from ipmn_proflow.imports import *


def net_info_1(dataset):
    df = dataset
    outgoing_count = df['Sender_account'].value_counts()
    incoming_count = df['Receiver_account'].value_counts()

    def calculate_counts(row):
        # add each account's transaction times together
        total_sender_count = (outgoing_count.get(row['Sender_account'], 0) +
                              incoming_count.get(row['Sender_account'], 0))
        # means sender user has How many transactions were made within the dataset, both in and out
        total_receiver_count = (outgoing_count.get(row['Receiver_account'], 0) +
                                incoming_count.get(row['Receiver_account'], 0))
        # means receive user has How many transactions were made within the dataset, both in and out

        return pd.Series({
            'out_same_count': total_sender_count,
            'in_same_count': total_receiver_count
        })

    df[['out_same_count', 'in_same_count']] = df.apply(calculate_counts, axis=1)

    return df
