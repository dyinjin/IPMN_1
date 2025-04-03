from imports import *


class UnitDataLoader:
    """
    A class to handle loading datasets through various methods.
    """

    @staticmethod
    def dataloader_year_month(config, year, month):
        """
        Load dataset for the specified year and month.

        Args:
            config (Config): The configuration object containing settings and paths.
            year (int): Year of the dataset.
            month (int): Month of the dataset.

        Returns:
            pd.DataFrame: The dataset loaded as a Pandas DataFrame with standardized columns.
        """
        # Construct the file path based on the year and month
        file_path = f'{config.DATAPATH}{year:04d}-{month:02d}.csv'

        # Check if the file exists
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")

            # Create an empty DataFrame with the standardized column structure
            data_set = pd.DataFrame(data=None, columns=config.STANDARD_INPUT_PARAM)

            # Read the CSV file into a temporary DataFrame
            csv_data = pd.read_csv(file_path)

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
    def dataloader_all(config):
        file_path = f'{config.DATAPATH}{config.ORI_ALL_CSV}'
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")

            # Create an empty DataFrame with the standardized column structure
            data_set = pd.DataFrame(data=None, columns=config.STANDARD_INPUT_PARAM)

            # Read the CSV file into a temporary DataFrame
            csv_data = pd.read_csv(file_path)

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

            # Calculate the cutoff date (two months after the earliest date)
            cutoff_date = earliest_date + pd.DateOffset(months=month)

            # Filter data to include only records within the first two months
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

        return data_set

