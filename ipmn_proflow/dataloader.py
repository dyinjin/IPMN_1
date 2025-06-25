from imports import *


class UnitDataLoader:
    """
    A class to handle loading datasets through various methods.
    """
    @staticmethod
    def dataloader_all(config):
        file_path = f'{config.DATAPATH}{config.ORI_ALL_CSV}'
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")

            # Create an empty DataFrame with the standardized column structure
            data_set = pd.DataFrame(data=None, columns=config.STANDARD_INPUT_PARAM)

            # Read the CSV file into a temporary DataFrame
            csv_data = pd.read_csv(file_path)

            # Convert the 'Date' column to datetime
            csv_data['Date'] = pd.to_datetime(csv_data['Date'])

            # Sort the data by 'Date' to ensure chronological order
            csv_data = csv_data.sort_values(by='Date')

            # Populate the standardized columns with data from the CSV file
            for column in data_set.columns:
                if column in csv_data.columns:
                    data_set[column] = csv_data[column]

            # Return the populated standardized dataset
            return data_set
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"'{file_path}' not found. Please check the path.")

    @staticmethod
    def dataloader_first(config, month):
        file_path = f'{config.DATAPATH}{config.ORI_ALL_CSV}'
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")

            # Create an empty DataFrame with the standardized column structure
            data_set = pd.DataFrame(data=None, columns=config.STANDARD_INPUT_PARAM)

            # Read the CSV file into a temporary DataFrame
            csv_data = pd.read_csv(file_path)

            # Convert the 'Date' column to datetime
            csv_data['Date'] = pd.to_datetime(csv_data['Date'])

            # Sort the data by 'Date' to ensure chronological order
            csv_data = csv_data.sort_values(by='Date')

            # Identify the earliest date in the dataset
            earliest_date = csv_data['Date'].min()

            # Calculate the cutoff date ('two' months after the earliest date)
            cutoff_date = earliest_date + pd.DateOffset(months=month)

            # Filter data to include only records within the first 'two' months
            csv_data = csv_data[(csv_data['Date'] >= earliest_date) & (csv_data['Date'] < cutoff_date)]

            # Populate the standardized columns with data from the filtered DataFrame
            for column in data_set.columns:
                if column in csv_data.columns:
                    data_set[column] = csv_data[column]

            # Return the populated standardized dataset
            return data_set
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"'{file_path}' not found. Please check the path.")

    @staticmethod
    def dataloader_between(config, start_date, end_date):
        file_path = f'{config.DATAPATH}{config.ORI_ALL_CSV}'
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")

            # Create an empty DataFrame with the standardized column structure
            data_set = pd.DataFrame(data=None, columns=config.STANDARD_INPUT_PARAM)

            # Read the CSV file into a temporary DataFrame
            csv_data = pd.read_csv(file_path)

            # Convert the 'Date' column to datetime
            csv_data['Date'] = pd.to_datetime(csv_data['Date'])

            # Sort the data by 'Date' to ensure chronological order
            csv_data = csv_data.sort_values(by='Date')

            # Convert start_date and end_date to datetime for filtering
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Filter data to include only records within the specified date range
            csv_data = csv_data[(csv_data['Date'] >= start_date) & (csv_data['Date'] <= end_date)]

            # Populate the standardized columns with data from the filtered DataFrame
            for column in data_set.columns:
                if column in csv_data.columns:
                    data_set[column] = csv_data[column]

            # clear
            del csv_data

            # Return the populated standardized dataset
            return data_set
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"'{file_path}' not found. Please check the path.")

    @staticmethod
    def csvloader_specified(config, csv_specified):
        file_path = f'{config.DATAPATH}{csv_specified}'
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")
            # Read the CSV file into a temporary DataFrame
            csv_data = pd.read_csv(file_path)
            # Data structures are not guaranteed to be available
            # need use specified uniter later
            return csv_data
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"'{file_path}' not found. Please check the path.")


    @staticmethod
    def datauniter_ibm(config, csv_data):
        # Create an empty DataFrame with the standardized column structure
        # STANDARD_INPUT_PARAM = ['Is_laundering',
        #                         'Date', 'Time', 'Sender_account', 'Receiver_account', 'Amount',
        #                         'Payment_currency', 'Received_currency', 'Payment_type']
        # IBM
        # 'Timestamp'
        # 'From Bank'	'Account'
        # 'To Bank'	'Account'
        # 'Amount Received'	'Receiving Currency'
        # 'Amount Paid'	'Payment Currency'
        # 'Payment Format'
        # 'Is Laundering'
        # TODO: Element consistency (SAML:UK pounds == IBM:UK Pound)
        data_set = pd.DataFrame(data=None, columns=config.STANDARD_INPUT_PARAM)
        data_set['Is_laundering'] = csv_data['Is Laundering']

        csv_data['Timestamp'] = pd.to_datetime(csv_data['Timestamp'])
        data_set['Date'] = csv_data['Timestamp'].dt.date
        data_set['Time'] = csv_data['Timestamp'].dt.time

        csv_data['Sender_account'] = csv_data['From Bank'].astype(str) + csv_data['Account']
        data_set['Sender_account'] = csv_data['Sender_account']

        csv_data['Receiver_account'] = csv_data['To Bank'].astype(str) + csv_data['Account.1']
        data_set['Receiver_account'] = csv_data['Receiver_account']

        # 'Amount Received' and	'Amount Paid' almost same just use Paid data
        data_set['Amount'] = csv_data['Amount Paid']

        data_set['Payment_currency'] = csv_data['Payment Currency']
        data_set['Received_currency'] = csv_data['Receiving Currency']

        data_set['Payment_type'] = csv_data['Payment Format']

        # Convert the 'Date' column to datetime
        data_set['Date'] = pd.to_datetime(data_set['Date'])

        # Sort the data by 'Date' to ensure chronological order
        data_set = data_set.sort_values(by='Date')

        return data_set


    @staticmethod
    def datauniter_saml(config, csv_data):
        data_set = pd.DataFrame(data=None, columns=config.STANDARD_INPUT_PARAM)

        # Convert the 'Date' column to datetime
        data_set['Date'] = pd.to_datetime(data_set['Date'])

        # Sort the data by 'Date' to ensure chronological order
        data_set = data_set.sort_values(by='Date')

        # Populate the standardized columns with data from the CSV file
        for column in data_set.columns:
            if column in csv_data.columns:
                data_set[column] = csv_data[column]

        # Return the populated standardized dataset
        return data_set


def load_dataset(config):
    # Load dataset based on mode specified in arguments
    """
    """
    print(f"Load dataset by mode: {config.DATASET_MODES}")

    if config.DATASET_MODES == config.MODE_QUICK_TEST:
        train_set = UnitDataLoader.dataloader_between(config, config.QT_TRAIN_START, config.QT_TRAIN_END)
        test_set = UnitDataLoader.dataloader_between(config, config.QT_TEST_START, config.QT_TEST_END)

    elif config.DATASET_MODES == config.MODE_ALL_D73:
        data_set = UnitDataLoader.dataloader_all(config)
        split_index = int(len(data_set) * 0.7)
        train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif config.DATASET_MODES == config.MODE_ALL_D82:
        data_set = UnitDataLoader.dataloader_all(config)
        split_index = int(len(data_set) * 0.8)
        train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif config.DATASET_MODES == config.MODE_FIRST_2_D73:
        data_set = UnitDataLoader.dataloader_first(config, 2)
        split_index = int(len(data_set) * 0.7)
        train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif config.DATASET_MODES == config.MODE_FIRST_4_D73:
        data_set = UnitDataLoader.dataloader_first(config, 4)
        split_index = int(len(data_set) * 0.7)
        train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif config.DATASET_MODES == config.MODE_IBM_D73:
        # TODO: ONLY IBM_d73 for BETA TEST
        # Thought the time span was too short(1day), NOT SUITABLE as a training set
        data_set = UnitDataLoader.csvloader_specified(config, config.IBM_CSV)
        data_set = UnitDataLoader.datauniter_ibm(config, data_set)
        # by random?
        train_set, test_set = train_test_split(data_set, test_size=0.3, random_state=config.RANDOM_SEED)
        train_set = train_set.sort_values(by='Date')
        test_set = test_set.sort_values(by='Date')
        # by time?
        # split_index = int(len(data_set) * 0.7)
        # train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif config.DATASET_MODES == config.MODE_SPECIFIC_TEST:
        # need different "datauniter" function, the column names of the dataset need be consistent
        train_set = UnitDataLoader.csvloader_specified(config, config.SP_TRAIN_FILE)
        train_set = UnitDataLoader.datauniter_saml(config, train_set)
        test_set = UnitDataLoader.csvloader_specified(config, config.SP_TEST_FILE)
        test_set = UnitDataLoader.datauniter_saml(config, test_set)

    else:
        # Raise an error if dataset mode is unsupported
        raise AttributeError(f"Dataset mode '{config.DATASET_MODES}' is not supported.")

    # reset index this make sure transaction match with each other
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    return train_set, test_set
