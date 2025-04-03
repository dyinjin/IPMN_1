import pandas as pd


def date_apart(dataset):
    """
    Splits date and time columns into individual components, adds a 'Timestamp' column,
    and replaces the original 'Time' column.

    Args:
        dataset (pd.DataFrame): Input dataset containing 'Date' and 'Time' columns.

    Returns:
        pd.DataFrame: Updated dataset with separate columns for year, month, week, day, hour, minute, second,
                     and a combined 'Timestamp' column.
    """
    # Ensure 'Date' and 'Time' columns are in datetime format
    if not pd.api.types.is_datetime64_any_dtype(dataset['Date']):
        dataset['Date'] = pd.to_datetime(dataset['Date'])  # Example format: '2022-10-07'
    if not pd.api.types.is_datetime64_any_dtype(dataset['Time']):
        dataset['Time'] = pd.to_datetime(dataset['Time'], format='%H:%M:%S')  # Example format: '10:35:19'

    # Extract components from 'Date' and 'Time' using vectorized operations
    dataset['Year'] = dataset['Date'].dt.year.astype('int16')
    dataset['Month'] = dataset['Date'].dt.month.astype('int8')
    dataset['Week'] = dataset['Date'].dt.isocalendar().week.astype('int8')
    dataset['Day'] = dataset['Date'].dt.day.astype('int8')
    dataset['Hour'] = dataset['Time'].dt.hour.astype('int8')
    dataset['Minute'] = dataset['Time'].dt.minute.astype('int8')
    dataset['Second'] = dataset['Time'].dt.second.astype('int8')

    # Combine 'Date' and 'Time' into a 'Timestamp' column
    dataset['Timestamp'] = pd.to_datetime(dataset['Date'].dt.date.astype(str) + " " +
                                          dataset['Time'].dt.time.astype(str))

    # Drop the original 'Time' column to save memory
    dataset.drop(columns=['Time'], inplace=True)

    return dataset
