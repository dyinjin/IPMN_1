from ipmn_proflow.imports import *


def main():
    """
    Main function
    """
    print("Program Start")
    config = Config()
    args = config.parse_arguments()

    # load dataset
    print(f"Load dataset by mode: {args.dataset}")
    if args.dataset == config.DATASET_MODES['quick_test']:
        data_set = UnitDataLoader().dataloader_year_month(config, config.QT_TRAIN_YEAR, config.QT_TRAIN_MONTH)
        print("Data set loaded successfully.")
    else:
        raise AttributeError(f"Dataset mode '{args.dataset}' is not supported.")

    # parameter handle
    print(f"Parameter handle by mode: {args.param_h}")
    if args.param_h == config.PARAMETER_MODES['time_date_division']:
        data_set = date_apart(data_set)
    elif args.param_h == config.PARAMETER_MODES['tdd_net_info_1']:
        # time date division
        data_set = date_apart(data_set)
        # add more net information
        data_set = net_info_1(data_set)
    else:
        raise AttributeError(f"Parameter handle mode '{args.param_h}' is not supported.")

    # label separate
    X = data_set.drop(columns=config.STANDARD_INPUT_LABEL)
    Y = data_set[config.STANDARD_INPUT_LABEL]

    # data balance
    print(f"Train/Test balanced by mode: {args.balance}")
    if args.balance == config.BALANCE_MODES['random_73']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=config.RANDOM_SEED)
    else:
        raise AttributeError(f"Balance mode '{args.balance}' is not supported.")

    # TODO more config support
    # TODO model choice
    # print(f"Train a model like: {args.model}")

    # numerical and categorical feature process for classification model
    numerical_features = X.select_dtypes(exclude="object").columns
    categorical_features = X.select_dtypes(include="object").columns

    transformer = ColumnTransformer(transformers=[
        ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        ("RobustScaler", RobustScaler(), numerical_features)
    ], remainder="passthrough")

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    # gird search
    param_grid = config.PARAM_GRID

    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=4,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print("Best Parameters: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # if roc_auc
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probabilities)
    print("Test AUC: ", test_auc)

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

    # matrix heatmap
    desired_tpr = config.TPR
    closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]

    y_pred = (test_probabilities >= closest_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix at {desired_tpr * 100}% TPR')
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    fpr_cm = fp / (fp + tn)
    tpr_cm = tp / (tp + fn)

    print(f"False Positive Rate (FPR): {fpr_cm:.3f}")
    print(f"True Positive Rate (TPR): {tpr_cm:.3f}")

    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
