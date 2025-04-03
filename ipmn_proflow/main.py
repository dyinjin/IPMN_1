from imports import *


def main():
    """
    Main function to execute the program logic.
    """
    print("Program Start")
    config = Config()
    args = config.parse_arguments()

    # flag 1: dataset do not need to divide to train/test, test_set specified
    # flag 0: dataset need to divide/balance to train/test
    separate_set_flag = 0

    # Load dataset based on mode specified in arguments
    print(f"Load dataset by mode: {args.dataset}")
    if args.dataset == config.DATASET_MODES['quick_test']:
        # Load quick program_test dataset using predefined year and month
        data_set = UnitDataLoader().dataloader_year_month(config, config.QT_TRAIN_YEAR, config.QT_TRAIN_MONTH)
        print("Dataset loaded successfully.")
    elif args.dataset == config.DATASET_MODES['all']:
        data_set = UnitDataLoader().dataloader_all(config)
    elif args.dataset == config.DATASET_MODES['first_2']:
        data_set = UnitDataLoader().dataloader_first(config, 2)
    elif args.dataset == config.DATASET_MODES['first_4']:
        data_set = UnitDataLoader().dataloader_first(config, 4)
    elif args.dataset == config.DATASET_MODES['IBM']:
        data_set = UnitDataLoader().csvloader_specified(config, config.IBM_CSV)
        data_set = UnitDataLoader().datauniter_ibm(config, data_set)
    elif args.dataset == config.DATASET_MODES['all_and_IBM']:
        # no balance but define a specified test csv
        separate_set_flag = 1
        data_set = UnitDataLoader().dataloader_all(config)
        test_set = UnitDataLoader().csvloader_specified(config, config.IBM_CSV)
        test_set = UnitDataLoader().datauniter_ibm(config, test_set)

    elif args.dataset == config.DATASET_MODES['IBM_and_first_2']:
        separate_set_flag = 1
        data_set = UnitDataLoader().csvloader_specified(config, config.IBM_CSV)
        data_set = UnitDataLoader().datauniter_ibm(config, data_set)
        test_set = UnitDataLoader().dataloader_first(config, 2)
    # TODO: Add support for more configuration options
    else:
        # Raise an error if dataset mode is unsupported
        raise AttributeError(f"Dataset mode '{args.dataset}' is not supported.")

    def parameter_adder(param_arg, dataset):
        if param_arg == config.PARAMETER_MODES['time_date_division']:
            # Split dataset by time and date
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_1']:
            # Perform time-date division
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            # Add additional network information
            dataset = net_info_tic(dataset)
            print("PARAMETER ADDED: transaction time counter")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_2']:
            # Perform time-date division
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            # Add additional network information
            dataset = net_info_tic(dataset)
            print("PARAMETER ADDED: transaction time counter")
            dataset = net_info_rtw(dataset)
            print("PARAMETER ADDED: recently trans with other account")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_3']:
            # Perform time-date division
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            # Add additional network information
            dataset = net_info_tic(dataset)
            print("PARAMETER ADDED: transaction time counter")
            dataset = net_info_rtw(dataset)
            print("PARAMETER ADDED: recently trans with other account")
            # TODO NEW PARAMETER: recently trans with other account gap how long?
            dataset = net_info_3centrality(dataset)
            print("PARAMETER ADDED: three kinds of graph centrality")
        # TODO: Add support for more configuration options
        else:
            # Raise an error if parameter handle mode is unsupported
            raise AttributeError(f"Parameter handle mode '{args.param_h}' is not supported.")
        return dataset

    # Handle parameters based on mode specified in arguments
    print(f"Parameter handle by mode: {args.param_h}")
    data_set = parameter_adder(args.param_h, data_set)

    # some account info mixed number and character
    for column in data_set.select_dtypes(include=['object']).columns:
        data_set[column] = data_set[column].astype(str)

    # Perform data balancing based on mode specified in arguments
    print(f"Train/Test balanced by mode: {args.balance}")
    if separate_set_flag:
        # flag first detect. no balance need. do data_set thing again to test_set
        print(f"Parameter in test handle by mode: {args.param_h}")
        test_set = parameter_adder(args.param_h, test_set)
        # some account info mixed number and character
        for column in test_set.select_dtypes(include=['object']).columns:
            test_set[column] = test_set[column].astype(str)

        X_train = test_set.drop(columns=config.STANDARD_INPUT_LABEL)
        y_train = test_set[config.STANDARD_INPUT_LABEL]
        X_test = test_set.drop(columns=config.STANDARD_INPUT_LABEL)
        y_test = test_set[config.STANDARD_INPUT_LABEL]
    elif args.balance == config.BALANCE_MODES['random_7_train_3_test']:
        # Separate features (X) and labels (y)
        X = data_set.drop(columns=config.STANDARD_INPUT_LABEL)
        y = data_set[config.STANDARD_INPUT_LABEL]
        # Split dataset into training and testing sets with a 70-30 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=config.RANDOM_SEED)
    elif args.balance == config.BALANCE_MODES['one_train_one_test']:
        # must load dataset contain at least two months
        X_train, X_test, y_train, y_test = CustomBalance.one_train_one_test(data_set, config)
    elif args.balance == config.BALANCE_MODES['rest_train_one_test']:
        # must load dataset contain at least two months
        X_train, X_test, y_train, y_test = CustomBalance.rest_train_one_test(data_set, config)
        # TODO: Add support for more configuration options
    else:
        # Raise an error if balance mode is unsupported
        raise AttributeError(f"Balance mode '{args.balance}' is not supported.")

    # TODO: Implement model choice functionality

    # already divide in week day hour etc.
    X_train.drop(columns=['Date', 'Timestamp'], inplace=True)

    # Process numerical and categorical features for classification models
    numerical_features = X_train.select_dtypes(exclude="object").columns
    categorical_features = X_train.select_dtypes(include="object").columns

    # "account" always categorical
    account_columns = [col for col in numerical_features if "account" in col]
    numerical_features = numerical_features.difference(account_columns)
    categorical_features = categorical_features.union(account_columns)

    transformer = ColumnTransformer(transformers=[
        # Encode categorical features using OrdinalEncoder
        ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        # Scale numerical features using RobustScaler
        ("RobustScaler", RobustScaler(), numerical_features)
    ], remainder="passthrough")

    # Apply transformations to training and testing datasets
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    # Perform hyperparameter tuning using grid search
    param_grid = config.PARAM_GRID
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=2,
        verbose=2
    )

    # Train the model with grid search
    grid_search.fit(X_train, y_train)
    print("Best Parameters: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate the model using ROC-AUC score
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probabilities)
    print("Test AUC: ", test_auc)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, test_probabilities)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % test_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Generate confusion matrix heatmap at the desired TPR threshold
    desired_tpr = config.TPR
    closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]

    # Predict labels based on threshold
    y_pred = (test_probabilities >= closest_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix at {desired_tpr * 100}% TPR')
    plt.show()

    # Extract confusion matrix metrics
    tn, fp, fn, tp = cm.ravel()
    fpr_cm = fp / (fp + tn)
    tpr_cm = tp / (tp + fn)

    # Print evaluation metrics
    print(f"False Positive Rate (FPR): {fpr_cm:.3f}")
    print(f"True Positive Rate (TPR): {tpr_cm:.3f}")

    # Print classification report for detailed model evaluation
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
