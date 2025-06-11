from imports import *

# for roc click monitor needed. store predict output of test_set
y_pred = None
# for test and avoid retraining
best_model = None

# TODO store functions in different files?


def load_config():
    """
    """
    print("Loading Config")
    config = Config()
    # args = config.parse_arguments()
    return config


def load_dataset(config):
    # Load dataset based on mode specified in arguments
    """
    """
    args = config.parse_arguments()
    print(f"Load dataset by mode: {args.dataset}")

    if args.dataset == config.DATASET_MODES['quick_test']:
        train_set = UnitDataLoader.dataloader_between(config, config.QT_TRAIN_START, config.QT_TRAIN_END)
        test_set = UnitDataLoader.dataloader_between(config, config.QT_TEST_START, config.QT_TEST_END)

    elif args.dataset == config.DATASET_MODES['all_d73']:
        data_set = UnitDataLoader.dataloader_all(config)
        split_index = int(len(data_set) * 0.7)
        train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif args.dataset == config.DATASET_MODES['first_2_d73']:
        data_set = UnitDataLoader.dataloader_first(config, 2)
        split_index = int(len(data_set) * 0.7)
        train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif args.dataset == config.DATASET_MODES['first_4_d73']:
        data_set = UnitDataLoader.dataloader_first(config, 4)
        split_index = int(len(data_set) * 0.7)
        train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif args.dataset == config.DATASET_MODES['IBM_d73']:
        # TODO: ONLY IBM_d73 for BETA TEST
        # Thought the time span was too short(1day), NOT SUITABLE as a training set
        data_set = UnitDataLoader.csvloader_specified(config, config.IBM_CSV)
        data_set = UnitDataLoader.datauniter_ibm(config, data_set)
        # by random?
        train_set, test_set = train_test_split(data_set, test_size=0.3, random_state=config.RANDOM_SEED)
        train_set = train_set.sort_values(by='Date')
        test_set = test_set.sort_values(by='Date')
        # by time?
        # split_index = int(len(data_set) * 0.7)
        # train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]

    elif args.dataset == config.DATASET_MODES['specific_train_specific_test']:
        # TODO: load name set in json config
        # need different "datauniter" function, the column names of the dataset need be consistent
        train_set = UnitDataLoader.csvloader_specified(config, "2022-11.csv")
        train_set = UnitDataLoader.datauniter_saml(config, train_set)
        test_set = UnitDataLoader.csvloader_specified(config, "2023-06.csv")
        test_set = UnitDataLoader.datauniter_saml(config, test_set)

    # TODO: Add support for more configuration options (2:8 split, )
    else:
        # Raise an error if dataset mode is unsupported
        raise AttributeError(f"Dataset mode '{args.dataset}' is not supported.")

    # reset index this make sure transaction match with each other
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    return train_set, test_set


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


def add_parameter(config, X_train, X_test):
    """
    """
    def parameter_adder(param_arg, dataset):
        if param_arg == config.PARAMETER_MODES['param_0']:
            # Split dataset by time and date
            # Default parameter
            dataset = date_apart(dataset)
        elif param_arg == config.PARAMETER_MODES['param_1']:
            dataset = date_apart(dataset)
            dataset = net_info_tic(dataset)
            print("PARAMETER ADDED: transaction time counter")
        elif param_arg == config.PARAMETER_MODES['param_2']:
            dataset = date_apart(dataset)
            dataset = net_info_rti(dataset)
            print("PARAMETER ADDED: recent association transaction info")
        elif param_arg == config.PARAMETER_MODES['param_3']:
            dataset = date_apart(dataset)
            dataset = net_info_3centrality(dataset)
            print("PARAMETER ADDED: three graph features")
        elif param_arg == config.PARAMETER_MODES['param_4']:
            dataset = date_apart(dataset)
            dataset = window_before(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: window cut association transaction info")
        elif param_arg == config.PARAMETER_MODES['param_5']:
            dataset = date_apart(dataset)
            dataset = window_before_graph(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: window cut graph features")
        elif param_arg == config.PARAMETER_MODES['param_6']:
            dataset = date_apart(dataset)
            dataset = window_slider(dataset, config.WINDOW_SIZE, config.SLIDER_STEP)
            print("PARAMETER ADDED: window slide association transaction info")
        elif param_arg == config.PARAMETER_MODES['param_7']:
            dataset = date_apart(dataset)
            dataset = window_slider_graph(dataset, config.WINDOW_SIZE, config.SLIDER_STEP)
            print("PARAMETER ADDED: window slide graph features")
        elif param_arg == config.PARAMETER_MODES['param_8']:
            dataset = date_apart(dataset)
            dataset = strict_before_(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: strict before association transaction info")
        elif param_arg == config.PARAMETER_MODES['param_9']:
            # not recommend
            dataset = date_apart(dataset)
            dataset = strict_before_graph_(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: strict before graph features")
        elif param_arg == config.PARAMETER_MODES['param_a']:
            dataset = date_apart(dataset)
            dataset = window_before_inte(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: window cut features")
        elif param_arg == config.PARAMETER_MODES['param_b']:
            dataset = date_apart(dataset)
            dataset = window_slider_inte(dataset, config.WINDOW_SIZE, config.SLIDER_STEP)
            print("PARAMETER ADDED: window slide features")
        # TODO: Add support for more configuration options
        else:
            # Raise an error if parameter handle mode is unsupported
            raise AttributeError(f"Parameter handle mode '{args.param}' is not supported.")
        return dataset

    # Handle parameters based on mode specified in arguments
    args = config.parse_arguments()
    print(f"Parameter handle by mode: {args.param}")
    X_train = parameter_adder(args.param, X_train)
    X_test = parameter_adder(args.param, X_test)

    # already divide in week day hour etc.
    X_train = X_train.drop(columns=config.STANDARD_TIME_PARAM)
    X_test = X_test.drop(columns=config.STANDARD_TIME_PARAM)

    # show final train/test columns
    # print(X_train.columns)
    # print(X_test.columns)
    return X_train, X_test


def save_feature_data2csv(config, y_train, y_test, X_train, X_test):
    args = config.parse_arguments()
    if config.SAVE_LEVEL == 1:
        # save train/test X/y to csv
        X_train.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_train.csv", index=False)
        X_test.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test.csv", index=False)
        y_train.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_train.csv", index=False)
        y_test.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_test.csv", index=False)
    elif config.SAVE_LEVEL == 2:
        # save train/test set to csv
        pd.concat([X_train, y_train], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_train_with_y.csv", index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test_with_y.csv", index=False)
    else:
        pass


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


def config_model(config):
    # TODO model config by args, XGBoost has many useful config
    xgb = XGBClassifier(eval_metric='logloss', random_state=config.RANDOM_SEED)
    param_grid = config.PARAM_GRID
    grid_search_model = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=2, verbose=2)
    return grid_search_model


def train_model(grid_search_model, X_train, y_train):
    # Train the model with grid search
    trained_grid_search_model = grid_search_model.fit(X_train, y_train.values.ravel())
    return trained_grid_search_model


def search_best(config, trained_grid_search_model, model_id):
    # TODO: find if any other method for best_estimator
    print("Best Parameters: ", trained_grid_search_model.best_params_)
    best_model = trained_grid_search_model.best_estimator_

    # 存储模型
    if config.SAVE_LEVEL != -1:
        model_path = f"{config.DATAPATH}{model_id}_{config.SAVE_MODEL}"
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

    return best_model


def analysis_importance(best_model, columns_name):
    # TODO: show/not by config
    # 获取特征重要性
    feature_importances = best_model.feature_importances_
    # 创建 DataFrame，使特征名称与重要性对应
    importance_df = pd.DataFrame({
        'Feature': columns_name,  # 使用预存的列名
        'Importance': feature_importances
    })
    # 按重要性排序
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()  # 让重要性最高的特征在顶部
    plt.show()
    # 输出特征重要性数据
    print(importance_df)


def test_model(best_model, X_test):
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    return test_probabilities


def save_predict_data2csv_float(config, test_probabilities, X_test, y_test, columns_name):
    args = config.parse_arguments()
    if config.SAVE_LEVEL == 0:
        pass
    elif config.SAVE_LEVEL == 1:
        pd.DataFrame(test_probabilities).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_prob.csv", index=False)
    elif config.SAVE_LEVEL == 2:
        test_probabilities = pd.Series(test_probabilities, name="predict_fraud_probability")
        pd.concat([pd.DataFrame(X_test, columns=columns_name), y_test, test_probabilities], axis=1).to_csv(
            f"{config.DATAPATH}{args.dataset}-{args.param}-X_test_with_prob.csv", index=False)


def analysis_performance(config, y_test, test_probabilities):
    global y_pred

    # Evaluate the model using ROC-AUC score
    test_auc = roc_auc_score(y_test, test_probabilities)
    print("Test AUC: ", test_auc)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, test_probabilities)

    # show curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    # set tpr for evaluate
    if config.TPR_SET == 0:
        # use preset TPR
        desired_tpr = config.TPR
        closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]
        y_pred = (test_probabilities >= closest_threshold).astype(int)
        print(f"Using preset TPR: {desired_tpr}, Closest Threshold: {closest_threshold}")

    elif config.TPR_SET == 1:
        # use mouse click position
        def onclick(event):
            global y_pred
            if event.xdata is not None and event.ydata is not None:
                desired_tpr = event.ydata  # 用户选择的 TPR 值
                closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]

                # 根据阈值计算预测标签
                y_pred = (test_probabilities >= closest_threshold).astype(int)
                print(f"Selected TPR: {desired_tpr}, Closest Threshold: {closest_threshold}")

        # monitor click
        fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix')
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


def save_predict_data2csv_bool(config, X_test, y_test, columns_name):
    global y_pred
    args = config.parse_arguments()
    if config.SAVE_LEVEL == 0:
        pass
    elif config.SAVE_LEVEL == 1:
        pd.DataFrame(y_pred).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_pred.csv", index=False)
    elif config.SAVE_LEVEL == 2:
        y_pred = pd.Series(y_pred, name="predict_fraud")
        pd.concat([pd.DataFrame(X_test, columns=columns_name), y_test, y_pred], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test_with_pred.csv", index=False)


def main():
    """
    Main function to execute the program logic.
    """

    global best_model

    print("Program Start")

    config = load_config()

    train_set, test_set = load_dataset(config)

    y_train, y_test, X_train, X_test = split_label(config, train_set, test_set)

    X_train, X_test = add_parameter(config, X_train, X_test)

    save_feature_data2csv(config, y_train, y_test, X_train, X_test)

    columns_name, X_train, X_test, model_id = encode_feature(config, X_train, X_test)

    grid_search_model = config_model(config)

    trained_grid_search_model = train_model(grid_search_model, X_train, y_train)

    best_model = search_best(config, trained_grid_search_model, model_id)

    analysis_importance(best_model, columns_name)
    # TODO: show tree, separate importance, graph

    test_probabilities = test_model(best_model, X_test)

    save_predict_data2csv_float(config, test_probabilities, X_test, y_test, columns_name)

    analysis_performance(config, y_test, test_probabilities)

    save_predict_data2csv_bool(config, X_test, y_test, columns_name)


if __name__ == '__main__':
    main()
