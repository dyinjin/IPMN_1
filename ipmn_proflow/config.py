from ipmn_proflow.imports import *


class Config:
    def __init__(self):
        self.DATAPATH = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}\\data\\'
        self.RANDOM_SEED = 42

        # don't change STANDARD_PARAM make new dataset follow the rules
        self.STANDARD_INPUT_PARAM = ['Is_laundering',
                                     'Date', 'Time', 'Sender_account', 'Receiver_account', 'Amount',
                                     'Payment_currency', 'Received_currency', 'Payment_type',
                                     ]

        self.STANDARD_INPUT_LABEL = 'Is_laundering'

        self.DATASET_MODES = {
            'default': 'default',
            'quick_test': 'quick_test',
            'all': 'all',
        }

        self.PARAMETER_MODES = {
            'default': 'default',
            'time_date_division': 'time_date_division',
            'tdd_net_info_1': 'tdd_net_info_1',
        }

        self.BALANCE_MODES = {
            'default': 'default',
            'random_73': 'random_73',
            'cut_73': 'cut_73',
        }

        self.PARAM_GRID = {
            'max_depth': [16],
            'eta': [0.1],
        }

        self.TPR = 0.9

        self.QT_TRAIN_YEAR = 2022
        self.QT_TRAIN_MONTH = 10
        self.QT_TEST_YEAR = 2022
        self.QT_TEST_MONTH = 12

    def parse_arguments(self):
        """
        Parse command-line arguments.

        Returns:
            argparse.Namespace: Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="")

        parser.add_argument(
            '--dataset',
            type=str,
            choices=self.DATASET_MODES.values(),
            default=self.DATASET_MODES['quick_test'],
            help='Specify dataset mode.'
        )

        parser.add_argument(
            '--param_h',
            type=str,
            choices=self.PARAMETER_MODES.values(),
            default=self.PARAMETER_MODES['time_date_division'],
            help='Specify parameter handling mode.'
        )

        parser.add_argument(
            '--balance',
            type=str,
            choices=self.BALANCE_MODES.values(),
            default=self.BALANCE_MODES['random_73'],
            help='Specify balance mode.'
        )

        return parser.parse_args()
