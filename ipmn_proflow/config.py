from imports import *


class Config:
    # Default values
    DEFAULTS = {
        "DATAPATH": "E:/default_download/IPMN_pro/IPMN_1/data/",
        "ORI_ALL_CSV": "SAML-D.csv",
        "IBM_CSV": "sampled_IBM.csv",
        "SAVE_TRANS": "saved_transformer.pkl",
        "SAVE_MODEL": "saved_model.pkl",
        "RANDOM_SEED": 42,
        "SAVE_LEVEL": -1,
        "SHOW_LEVEL": 1,
        "STANDARD_INPUT_PARAM": [
            "Is_laundering",
            "Laundering_type",
            "Date",
            "Time",
            "Sender_account",
            "Receiver_account",
            "Amount",
            "Payment_currency",
            "Received_currency",
            "Payment_type"
        ],
        "STANDARD_INPUT_LABEL": "Is_laundering",
        "MULTI_CLASS_LABEL": "Laundering_type",
        "STANDARD_TIME_PARAM": [
            "Date",
            "Timestamp",
            "Year",
            "Month"
        ],
        "WINDOW_SIZE": 10,
        "SLIDER_STEP": 1,
        "PARAM_GRID": {
            "max_depth": [
                14,
                16
            ],
            "eta": [
                0.12,
                0.14
            ]
        },
        "TPR": 0.95,
        "TPR_SET": 0,
        "DATASET_MODES": "quick_test",
        "PARAMETER_MODES": "basic",
        "RUN_MODES": "basic",
        # quick_test mode need
        "QT_TRAIN_START": "2022/11/01",
        "QT_TRAIN_END": "2022/11/30",
        "QT_TEST_START": "2023/04/01",
        "QT_TEST_END": "2023/04/30",
        # specific_train_specific_test mode need
        "SP_TRAIN_FILE": "2022-11.csv",
        "SP_TEST_FILE": "2023-06.csv",
    }

    # Valid options for mode validation
    # Dataset modes
    MODE_QUICK_TEST = 'quick_test'
    MODE_ALL_D73 = 'all_d73'
    MODE_ALL_D82 = 'all_d82'
    MODE_FIRST_2_D73 = 'first_2_d73'
    MODE_FIRST_4_D73 = 'first_4_d73'
    MODE_IBM_D73 = 'IBM_d73'
    MODE_SPECIFIC_TEST = 'specific_train_specific_test'

    VALID_DATASET_MODES = [
        MODE_QUICK_TEST,
        MODE_ALL_D73,
        MODE_ALL_D82,
        MODE_FIRST_2_D73,
        MODE_FIRST_4_D73,
        MODE_IBM_D73,
        MODE_SPECIFIC_TEST,
    ]

    # Parameter handling modes
    PARAM_ORIGIN = 'origin'
    PARAM_BASIC = 'basic'
    PARAM_UNLIMITED_RECENT = 'unlimited_recent'
    PARAM_UNLIMITED_GRAPH = 'unlimited_graph'
    PARAM_ROLLING_ASSOC = 'rolling_association'
    PARAM_ROLLING_GRAPH = 'rolling_graph'
    PARAM_ROLLING_BOTH = 'rolling_both'
    PARAM_SLIDING_ASSOC = 'sliding_association'
    PARAM_SLIDING_GRAPH = 'sliding_graph'
    PARAM_SLIDING_BOTH = 'sliding_both'
    PARAM_TEST_0 = 'test_0'
    PARAM_TEST_1 = 'test_1'

    VALID_PARAMETER_MODES = [
        PARAM_ORIGIN, PARAM_BASIC,
        PARAM_UNLIMITED_RECENT, PARAM_UNLIMITED_GRAPH,
        PARAM_ROLLING_ASSOC, PARAM_ROLLING_GRAPH, PARAM_ROLLING_BOTH,
        PARAM_SLIDING_ASSOC, PARAM_SLIDING_GRAPH, PARAM_SLIDING_BOTH,
        PARAM_TEST_0, PARAM_TEST_1
    ]

    # Run modes
    RUN_BASIC = 'basic'
    RUN_MULTI_CLASS = 'multi-class'

    VALID_RUN_MODES = [RUN_BASIC, RUN_MULTI_CLASS]

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)

        # Warn about unexpected keys in JSON
        for key in user_config:
            if key not in self.DEFAULTS:
                warnings.warn(f"[Config Warning] Unexpected key in JSON: '{key}' — ignored")

        # Warn about missing expected keys
        for key in self.DEFAULTS:
            if key not in user_config:
                warnings.warn(f"[Config Warning] Missing key in JSON: '{key}' — using default: {self.DEFAULTS[key]}")

        # Merge user config over defaults
        merged_config = {**self.DEFAULTS, **{k: v for k, v in user_config.items() if k in self.DEFAULTS}}

        for key, value in merged_config.items():
            setattr(self, key, value)

        # Validate selected modes
        if self.DATASET_MODES not in self.VALID_DATASET_MODES:
            raise ValueError(f"Invalid DATASET_MODES: '{self.DATASET_MODES}'. Valid: {self.VALID_DATASET_MODES}")
        if self.PARAMETER_MODES not in self.VALID_PARAMETER_MODES:
            raise ValueError(f"Invalid PARAMETER_MODES: '{self.PARAMETER_MODES}'. Valid: {self.VALID_PARAMETER_MODES}")
        if self.RUN_MODES not in self.VALID_RUN_MODES:
            raise ValueError(f"Invalid RUN_MODES: '{self.RUN_MODES}'. Valid: {self.VALID_RUN_MODES}")

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Load config from JSON file.")
        parser.add_argument('--config_path', type=str, required=True, help='Path to configuration JSON file')
        return parser.parse_args()


def load_config():
    """
    """
    print("Loading Config")
    args = Config.parse_arguments()
    config = Config(args.config_path)

    return config