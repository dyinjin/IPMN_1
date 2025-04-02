from ipmn_proflow.imports import *


class Config:
    """
    Configuration class for managing dataset paths, parameters, modes, and hyperparameters.
    """

    def __init__(self):
        # Path to the data directory
        self.DATAPATH = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}\\data\\'

        self.ORI_ALL_CSV = 'SAML-D.csv'
        # Random seed for reproducibility
        self.RANDOM_SEED = 42

        # Standardized input parameters that datasets must follow
        self.STANDARD_INPUT_PARAM = ['Is_laundering',
                                     'Date', 'Time', 'Sender_account', 'Receiver_account', 'Amount',
                                     'Payment_currency', 'Received_currency', 'Payment_type']

        # Label for classification tasks
        self.STANDARD_INPUT_LABEL = 'Is_laundering'

        # Dataset modes for argument selection
        self.DATASET_MODES = {
            # 'default': 'default',
            'quick_test': 'quick_test',
            'all': 'all',
            'first_2': 'first_2',
            'first_4': 'first_4'
        }

        # Parameter handling modes
        self.PARAMETER_MODES = {
            # 'default': 'default',
            'time_date_division': 'time_date_division',
            'tdd_net_info_1': 'tdd_net_info_1',
        }

        # Data balancing modes for splitting datasets
        self.BALANCE_MODES = {
            # 'default': 'default',
            'random_73': 'random_73',
            # 'cut_73': 'cut_73',
            'one_one': 'one_one',
            'rest_one': 'rest_one'
        }

        # Hyperparameter grid for model tuning
        self.PARAM_GRID = {
            'max_depth': [16],
            'eta': [0.1],
        }

        # Target true positive rate (TPR) for model evaluation
        self.TPR = 0.9

        # Predefined quick test dataset time configuration for quick-test
        self.QT_TRAIN_YEAR = 2022
        self.QT_TRAIN_MONTH = 10
        self.QT_TEST_YEAR = 2022
        self.QT_TEST_MONTH = 12

        self.TRAIN_MONTH_OFFSET = 1
        self.TEST_MONTH_OFFSET = 4

    def parse_arguments(self):
        """
        Parse command-line arguments for dataset mode, parameter handling mode, and balance mode.

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
            '--param_h',
            type=str,
            choices=self.PARAMETER_MODES.values(),
            default=self.PARAMETER_MODES['time_date_division'],
            help='Specify parameter handling mode.'
        )

        # Data balancing mode argument
        parser.add_argument(
            '--balance',
            type=str,
            choices=self.BALANCE_MODES.values(),
            default=self.BALANCE_MODES['random_73'],
            help='Specify balance mode.'
        )

        return parser.parse_args()
