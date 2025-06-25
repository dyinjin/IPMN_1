from imports import *
from parameter_handler import *


def split_label(config, train_set, test_set):
    y_train = train_set[config.STANDARD_INPUT_LABEL]
    y_test = test_set[config.STANDARD_INPUT_LABEL]
    X_train = train_set.drop([config.STANDARD_INPUT_LABEL, config.MULTI_CLASS_LABEL], axis=1)
    X_test = test_set.drop([config.STANDARD_INPUT_LABEL, config.MULTI_CLASS_LABEL], axis=1)

    # TODO: MULTI_CLASS_LABEL multi-classification
    # y_train = train_set[config.MULTI_CLASS_LABEL]
    # y_test = test_set[config.MULTI_CLASS_LABEL]
    # from sklearn.preprocessing import OneHotEncoder
    # one_hot_encoder = OneHotEncoder(sparse_output=False)
    # y_train = one_hot_encoder.fit_transform(y_train.values.reshape(-1, 1))
    # y_test = one_hot_encoder.transform(y_test.values.reshape(-1, 1))
    # X_train = train_set.drop([config.STANDARD_INPUT_LABEL, config.MULTI_CLASS_LABEL], axis=1)
    # X_test = test_set.drop([config.STANDARD_INPUT_LABEL, config.MULTI_CLASS_LABEL], axis=1)

    print("train set laundering count:")
    print(y_train.value_counts())
    print("test set laundering count:")
    print(y_test.value_counts())

    return y_train, y_test, X_train, X_test


def parameter_adder(config, param_config, dataset):
    if param_config == config.PARAM_ORIGIN:
        # Split dataset by time and date
        # Default parameter
        dataset = date_apart(dataset)
    elif param_config == config.PARAM_BASIC:
        dataset = date_apart(dataset)
        dataset = net_info_tic(dataset)
        print("PARAMETER ADDED: transaction time counter")
    elif param_config == config.PARAM_UNLIMITED_RECENT:
        dataset = date_apart(dataset)
        dataset = net_info_rti(dataset)
        print("PARAMETER ADDED: recent association transaction info")
    elif param_config == config.PARAM_UNLIMITED_GRAPH:
        dataset = date_apart(dataset)
        dataset = net_info_3centrality(dataset)
        print("PARAMETER ADDED: three graph features")
    elif param_config == config.PARAM_ROLLING_ASSOC:
        dataset = date_apart(dataset)
        dataset = window_before(dataset, config.WINDOW_SIZE)
        print("PARAMETER ADDED: window cut association transaction info")
    elif param_config == config.PARAM_ROLLING_GRAPH:
        dataset = date_apart(dataset)
        dataset = window_before_graph(dataset, config.WINDOW_SIZE)
        print("PARAMETER ADDED: window cut graph features")
    elif param_config == config.PARAM_ROLLING_BOTH:
        dataset = date_apart(dataset)
        dataset = window_before_inte(dataset, config.WINDOW_SIZE)
        print("PARAMETER ADDED: window cut features")
    elif param_config == config.PARAM_SLIDING_ASSOC:
        dataset = date_apart(dataset)
        dataset = window_slider(dataset, config.WINDOW_SIZE, config.SLIDER_STEP)
        print("PARAMETER ADDED: window slide association transaction info")
    elif param_config == config.PARAM_SLIDING_GRAPH:
        dataset = date_apart(dataset)
        dataset = window_slider_graph(dataset, config.WINDOW_SIZE, config.SLIDER_STEP)
        print("PARAMETER ADDED: window slide graph features")
    elif param_config == config.PARAM_SLIDING_BOTH:
        dataset = date_apart(dataset)
        dataset = window_slider_inte(dataset, config.WINDOW_SIZE, config.SLIDER_STEP)
        print("PARAMETER ADDED: window slide features")
    elif param_config == config.PARAM_TEST_0:
        dataset = date_apart(dataset)
        dataset = strict_before_(dataset, config.WINDOW_SIZE)
        print("PARAMETER ADDED: strict before association transaction info")
    elif param_config == config.PARAM_TEST_1:
        # not recommend
        dataset = date_apart(dataset)
        dataset = strict_before_graph_(dataset, config.WINDOW_SIZE)
        print("PARAMETER ADDED: strict before graph features")
    # TODO: Add support for more configuration options
    else:
        # Raise an error if parameter handle mode is unsupported
        raise AttributeError(f"Parameter handle mode '{config.PARAMETER_MODES}' is not supported.")
    return dataset


def add_parameter(config, X_train, X_test):
    """
    """
    # Handle parameters based on mode specified in arguments
    print(f"Parameter handle by mode: {config.PARAMETER_MODES}")
    X_train = parameter_adder(config, config.PARAMETER_MODES, X_train)
    X_test = parameter_adder(config, config.PARAMETER_MODES, X_test)

    # already divide in week day hour etc.
    X_train = X_train.drop(columns=config.STANDARD_TIME_PARAM)
    X_test = X_test.drop(columns=config.STANDARD_TIME_PARAM)

    # show final train/test columns
    # print(X_train.columns)
    # print(X_test.columns)
    return X_train, X_test


def encode_feature(config, X_train, X_test):
    # Process numerical and categorical features for classification models
    numerical_features = X_train.select_dtypes(exclude="object").columns
    categorical_features = X_train.select_dtypes(include="object").columns

    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    transformer = ColumnTransformer(transformers=[
        # Encode categorical features using OrdinalEncoder
        ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        # Scale numerical features using RobustScaler
        ("RobustScaler", RobustScaler(), numerical_features)
    ], remainder="passthrough")

    # TODO: check ColumnTransformer before and after

    # Apply transformations to training and testing datasets
    columns_name = X_train.columns
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    # save column transformer
    model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.SAVE_LEVEL != -1:
        transformer_path = f"{config.DATAPATH}{model_id}_{config.SAVE_TRANS}"
        joblib.dump(transformer, transformer_path)
        print(f"Transformer saved to {transformer_path}")

    print("train set after column transformer shape:")
    print(X_train.shape)
    print("test set after column transformer shape:")
    print(X_test.shape)

    return columns_name, X_train, X_test, model_id
