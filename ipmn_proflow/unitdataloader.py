from ipmn_proflow.imports import *


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
    def dataloader_first_2(config):
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
            cutoff_date = earliest_date + pd.DateOffset(months=2)

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

