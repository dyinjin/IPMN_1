from ipmn_proflow.imports import *


class UnitDataLoader:
    """
    A class to handle loading datasets by different method.
    """

    @staticmethod
    def dataloader_year_month(config, year, month):
        """
        Load dataset for the specified year and month.

        Args:
            config
            year (int): Year of the dataset.
            month (int): Month of the dataset.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = f'{config.DATAPATH}{year:04d}-{month:02d}.csv'
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")
            data_set = pd.DataFrame(data=None, columns=config.STANDARD_INPUT_PARAM)
            csv_data = pd.read_csv(file_path)
            for column in data_set.columns:
                if column in csv_data.columns:
                    data_set[column] = csv_data[column]
            return data_set
        else:
            raise FileNotFoundError(f"'{file_path}' not found. Please check the path.")
