from ipmn_proflow.imports import *


def net_info_1(dataset):
    df = dataset
    outgoing_count = df['Sender_account'].value_counts()
    incoming_count = df['Receiver_account'].value_counts()

    def calculate_counts(row):
        total_sender_count = (outgoing_count.get(row['Sender_account'], 0) +
                              incoming_count.get(row['Sender_account'], 0))
        total_receiver_count = (outgoing_count.get(row['Receiver_account'], 0) +
                                incoming_count.get(row['Receiver_account'], 0))

        return pd.Series({
            'out_same_count': total_sender_count,
            'in_same_count': total_receiver_count
        })

    df[['out_same_count', 'in_same_count']] = df.apply(calculate_counts, axis=1)

    return df
