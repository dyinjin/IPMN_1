from imports import *


class Config:
    """
    Configuration class for managing dataset paths, parameters, modes, and hyperparameters.
    """

    def __init__(self):
        # Path to the data directory
        self.DATAPATH = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}/data/'

        self.ORI_ALL_CSV = 'SAML-D.csv'
        self.IBM_CSV = 'HI-Small_Trans.csv'
        # Random seed for reproducibility
        self.RANDOM_SEED = 42

        # Standardized input parameters that datasets must follow
        self.STANDARD_INPUT_PARAM = ['Is_laundering',
                                     'Date', 'Time', 'Sender_account', 'Receiver_account', 'Amount',
                                     'Payment_currency', 'Received_currency', 'Payment_type']

        # Label for classification tasks
        self.STANDARD_INPUT_LABEL = ['Is_laundering']

        self.STANDARD_TIME_PARAM = ['Date', 'Timestamp']

        # Dataset modes for argument selection
        self.DATASET_MODES = {
            # 'default': 'default',
            'quick_test': 'quick_test',
            'all': 'all',
            'first_2': 'first_2',  # first 60 days' data
            'first_4': 'first_4',  # first 120 days' data
            'IBM': 'IBM',
            'all_and_IBM': 'all_and_IBM',
            'IBM_and_first_2': 'IBM_and_first_2',
        }

        # Parameter handling modes
        self.PARAMETER_MODES = {
            # 'default': 'default',
            'time_date_division': 'time_date_division',
            # tdd stands for time_date_division
            'tdd_net_info_1': 'tdd_net_info_1',
            'tdd_net_info_2': 'tdd_net_info_2',
            'tdd_net_info_3': 'tdd_net_info_3',
        }

        # Data division modes for splitting datasets
        self.DIVISION_MODES = {
            # 'default': 'default',
            'random_7_train_3_test': 'random_7_train_3_test',
            'cut_7_train_3_test': 'cut_7_train_3_test',
            'one_train_one_test': 'one_train_one_test',  # one month(not 30 days) for train/test.
            # Specific offset define by TRAIN_MONTH_OFFSET, TEST_MONTH_OFFSET

            'rest_train_one_test': 'rest_train_one_test'
            # last month(not 30 days) data for test. rest data before the last month as train
        }

        self.BALANCE_MODES = {
            'default': 'default',
            'smote': 'smote',
        }

        # Hyperparameter grid for model tuning
        self.PARAM_GRID = {
            'max_depth': [16, 18],
            'eta': [0.1],
        }

        # Target true positive rate (TPR) for model evaluation
        self.TPR = 0.95

        # Predefined quick test dataset time configuration for quick-test
        self.QT_TRAIN_YEAR = 2022
        self.QT_TRAIN_MONTH = 10
        self.QT_TEST_YEAR = 2022
        self.QT_TEST_MONTH = 12

        self.TRAIN_MONTH_OFFSET = 1
        self.TEST_MONTH_OFFSET = 4

    def parse_arguments(self):
        """
        Parse command-line arguments for dataset mode, parameter handling mode, and divide mode.

        Returns:
            argparse.Namespace: Parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(description="Argument parser for configuring dataset and processing modes.")

        # Dataset mode argument
        parser.add_argument(
            '--dataset',
            type=str,
            choices=self.DATASET_MODES.values(),
            default=self.DATASET_MODES['quick_test'],
            help='Specify dataset mode.'
        )

        # Parameter handling mode argument
        parser.add_argument(
            '--param',
            type=str,
            choices=self.PARAMETER_MODES.values(),
            default=self.PARAMETER_MODES['time_date_division'],
            help='Specify parameter handling mode.'
        )

        # Data division mode argument
        parser.add_argument(
            '--division',
            type=str,
            choices=self.DIVISION_MODES.values(),
            default=self.DIVISION_MODES['random_7_train_3_test'],
            help='Specify division mode.'
        )

        # Data balance mode argument
        parser.add_argument(
            '--balance',
            type=str,
            choices=self.BALANCE_MODES.values(),
            default=self.BALANCE_MODES['default'],
            help='Specify balance mode.'
        )

        return parser.parse_args()
