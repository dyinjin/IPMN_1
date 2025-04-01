from ipmn_proflow.imports import *


def date_apart(dataset):
    df = dataset
    # date in format like 2022-10-07
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df['Second'] = df['Time'].dt.second

    df.drop(columns=['Date', 'Time'], inplace=True)
    return df
