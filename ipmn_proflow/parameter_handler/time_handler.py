from ipmn_proflow.imports import *


def date_apart(dataset):
    """
    Splits date and time columns into individual components and adds a 'Timestamp' column.

    Args:
        dataset (pd.DataFrame): Input dataset containing 'Date' and 'Time' columns.

    Returns:
        pd.DataFrame: Updated dataset with separate columns for year, month, week, day, hour, minute, second, and a combined 'Timestamp' column.
                     Original 'Date' and 'Time' columns are dropped.
    """
    # Make a copy of the dataset to avoid modifying the original data
    df = dataset

    # Convert 'Date' and 'Time' columns to datetime format
    df['Date'] = pd.to_datetime(df['Date'])  # Example format: '2022-10-07'
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')  # Example format: '10:35:19'

    # Extract components from 'Date'
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day

    # Extract components from 'Time'
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df['Second'] = df['Time'].dt.second

    # Drop original 'Date' and 'Time' columns
    # df.drop(columns=['Date', 'Time'], inplace=True)

    # Return the updated dataset
    return df
