from imports import *


class Config:
    """
    Configuration class for managing dataset paths, parameters, modes, and hyperparameters.
    """

    def __init__(self):
        # Path to the data directory
        self.DATAPATH = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}\\data\\'

        self.ORI_ALL_CSV = 'SAML-D.csv'

        self.IBM_CSV_H = 'sampled_IBM.csv'

        self.SAVE_TRANS = 'saved_transformer.pkl'
        self.SAVE_MODEL = 'saved_model.pkl'

        # Random seed for reproducibility
        self.RANDOM_SEED = 42

        self.SAVE_TRAIN_TEST = 2

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
            'all_d73': 'all_d73',
            'first_2_d73': 'first_2_d73',
            'first_4_d73': 'first_4_d73',
            'IBM_d73': 'IBM_d73',
            'specific_train_specific_test': 'specific_train_specific_test',
        }

        # Predefined quick test dataset time configuration
        self.QT_TRAIN_START = '2022/11/01'
        self.QT_TRAIN_END = '2022/11/30'
        self.QT_TEST_START = '2023/07/01'
        self.QT_TEST_END = '2023/07/31'

        # Parameter handling modes
        self.PARAMETER_MODES = {
            'param_0': 'param_0',
            'param_1': 'param_1',
            'param_2': 'param_2',
            'param_3': 'param_3',
            'param_4': 'param_4',
            'param_5': 'param_5',
            'param_6': 'param_6',
            'param_7': 'param_7',
            'param_a': 'param_a',
            'param_b': 'param_b',
        }

        self.WINDOW_SIZE = 7

        # Hyperparameter grid for XGBoost model tuning
        self.PARAM_GRID = {
            'max_depth': [16, 18],
            'eta': [0.1],
        }

        # Target true positive rate (TPR) for model evaluation
        self.TPR = 0.95

        self.TPR_SET = 0

        # program running modes
        self.RUN_MODES = {
            'param_0': 'param_0',
            'param_1': 'param_1',
            'param_2': 'param_2',
            'param_3': 'param_3',
        }

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
            default=self.PARAMETER_MODES['param_0'],
            help='Specify parameter handling mode.'
        )

        return parser.parse_args()
