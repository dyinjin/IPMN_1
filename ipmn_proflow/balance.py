from ipmn_proflow.imports import *
import pandas as pd


class CustomBalance:
    @staticmethod
    def get_month_data(dataset, month_start, month_end):
        """
        Filter the dataset for entries between the specified start and end dates.

        Parameters:
        - dataset: DataFrame containing the data.
        - month_start: Start date for filtering.
        - month_end: End date for filtering.

        Returns:
        - Filtered DataFrame.
        """
        # Assuming the Date column is already in datetime format
        return dataset.loc[(dataset['Date'] >= month_start) & (dataset['Date'] < month_end)]

    @staticmethod
    def one_train_one_test(data_set, config):
        """
        Create one training set and one test set based on the configured offsets.

        Parameters:
        - data_set: DataFrame containing the data, sorted by date.
        - config: Configuration object containing TRAIN_MONTH_OFFSET, TEST_MONTH_OFFSET,
                  and STANDARD_INPUT_LABEL.

        Returns:
        - X_train, X_test: Features for training and testing.
        - y_train, y_test: Labels for training and testing.
        """
        # Convert date column to datetime format if not already
        data_set['Date'] = pd.to_datetime(data_set['Date'])

        # Cache configuration attributes for efficiency
        train_offset = config.TRAIN_MONTH_OFFSET
        test_offset = config.TEST_MONTH_OFFSET
        label_column = config.STANDARD_INPUT_LABEL

        start_date = data_set['Date'].min()

        # Efficient date calculations using pandas offsets
        train_start = pd.Timestamp(start_date.year, start_date.month, 1) + pd.DateOffset(months=train_offset - 1)
        train_end = train_start + pd.DateOffset(months=1)
        test_start = pd.Timestamp(start_date.year, start_date.month, 1) + pd.DateOffset(months=test_offset - 1)
        test_end = test_start + pd.DateOffset(months=1)

        train_set = CustomBalance.get_month_data(data_set, train_start, train_end)
        test_set = CustomBalance.get_month_data(data_set, test_start, test_end)

        # Efficiently drop label column and use .loc[]
        X_train = train_set.drop(columns=[label_column])
        y_train = train_set.loc[:, label_column]
        X_test = test_set.drop(columns=[label_column])
        y_test = test_set.loc[:, label_column]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def rest_train_one_test(data_set, config):
        """
        Use all but the last month of the dataset as the training set, and the last month as the test set.

        Parameters:
        - data_set: DataFrame containing the data, sorted by date.
        - config: Configuration object containing STANDARD_INPUT_LABEL.

        Returns:
        - X_train, X_test: Features for training and testing.
        - y_train, y_test: Labels for training and testing.
        """
        # Convert date column to datetime format if not already
        data_set['Date'] = pd.to_datetime(data_set['Date'])

        # Cache configuration attributes for efficiency
        label_column = config.STANDARD_INPUT_LABEL

        start_date = data_set['Date'].min()
        last_date = data_set['Date'].max()

        last_month_start = pd.Timestamp(last_date.year, last_date.month, 1)
        next_month_start = last_month_start + pd.DateOffset(months=1)

        train_set = CustomBalance.get_month_data(data_set, start_date, last_month_start)
        test_set = CustomBalance.get_month_data(data_set, last_month_start, next_month_start)

        X_train = train_set.drop(columns=[label_column])
        y_train = train_set.loc[:, label_column]
        X_test = test_set.drop(columns=[label_column])
        y_test = test_set.loc[:, label_column]

        return X_train, X_test, y_train, y_test
