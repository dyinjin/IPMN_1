from imports import *


def main():
    """
    Main function to execute the program logic.
    """
    print("Program Start")
    print("Loading Config")
    config = Config()
    args = config.parse_arguments()

    # Load dataset based on mode specified in arguments
    """
    Argument --dataset: Dataset Load mode
    """
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
        UnitDataLoader.csvloader_specified()
    # elif args.dataset == config.DATASET_MODES['one_train_one_test']:
    # elif args.dataset == config.DATASET_MODES['one_train_one_test']:
    # elif args.dataset == config.DATASET_MODES['all_train_IBM_test']:
    # elif args.dataset == config.DATASET_MODES['specific_train_specific_test']:
    # TODO: Add support for more configuration options
    else:
        # Raise an error if dataset mode is unsupported
        raise AttributeError(f"Dataset mode '{args.dataset}' is not supported.")

    # reset index
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    """
    Argument --param: Feature Parameter mode
    """
    def parameter_adder(param_arg, dataset):
        if param_arg == config.PARAMETER_MODES['tdd_net_info_0']:
            # Split dataset by time and date
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_1']:
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            dataset = net_info_tic(dataset)
            print("PARAMETER ADDED: transaction time counter")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_2']:
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            dataset = net_info_rti(dataset)
            print("PARAMETER ADDED: association transaction info")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_3']:
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            dataset = net_info_3centrality(dataset)
            print("PARAMETER ADDED: three graph features")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_4']:
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            dataset = net_info_before(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: before transaction info")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_5']:
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            dataset = net_info_before_with_graph(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: before transaction info with child graph")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_6']:
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            dataset = net_info_slider(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: before transaction info")
        elif param_arg == config.PARAMETER_MODES['tdd_net_info_7']:
            dataset = date_apart(dataset)
            print("PARAMETER ADDED: date/time parameter divide")
            dataset = net_info_slider_with_graph(dataset, config.WINDOW_SIZE)
            print("PARAMETER ADDED: before transaction info with child graph")
        # TODO: Add support for more configuration options
        else:
            # Raise an error if parameter handle mode is unsupported
            raise AttributeError(f"Parameter handle mode '{args.param}' is not supported.")
        return dataset

    y_train = train_set[config.STANDARD_INPUT_LABEL]
    y_test = test_set[config.STANDARD_INPUT_LABEL]
    X_train = train_set.drop(columns=config.STANDARD_INPUT_LABEL)
    X_test = test_set.drop(columns=config.STANDARD_INPUT_LABEL)

    # Handle parameters based on mode specified in arguments
    print(f"Parameter handle by mode: {args.param}")
    X_train = parameter_adder(args.param, X_train)
    X_test = parameter_adder(args.param, X_test)

    # already divide in week day hour etc.
    X_train = X_train.drop(columns=config.STANDARD_TIME_PARAM)
    X_test = X_test.drop(columns=config.STANDARD_TIME_PARAM)

    # show final train/test columns
    # print(X_train.columns)
    # print(X_test.columns)

    # TODO config save or not by args
    # save train/test X/y to csv
    X_train.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_train.csv", index=False)
    X_test.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test.csv", index=False)
    y_train.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_train.csv", index=False)
    y_test.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_test.csv", index=False)

    # # save train/test set to csv
    # pd.concat([X_train, y_train], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_train_with_y.csv", index=False)
    # pd.concat([X_test, y_test], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test_with_y.csv", index=False)

    # Process numerical and categorical features for classification models
    numerical_features = X_train.select_dtypes(exclude="object").columns
    categorical_features = X_train.select_dtypes(include="object").columns

    # "account" always categorical
    account_columns = [col for col in numerical_features if "account" in col]
    numerical_features = numerical_features.difference(account_columns)
    categorical_features = categorical_features.union(account_columns)

    # TODO: Implement model choice functionality
    transformer = ColumnTransformer(transformers=[
        # Encode categorical features using OrdinalEncoder
        ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        # Scale numerical features using RobustScaler
        ("RobustScaler", RobustScaler(), numerical_features)
    ], remainder="passthrough")

    # Apply transformations to training and testing datasets
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    print("train set laundering count:")
    print(y_train.value_counts())
    print("test set laundering count:")
    print(y_test.value_counts())

    print("train set transformer shape:")
    print(X_train.shape)
    print("test set transformer shape:")
    print(X_test.shape)


    # TODO model config by args
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    param_grid = config.PARAM_GRID
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=2, verbose=2)

    # rf = RandomForestClassifier()
    # param_grid = {"n_estimators": [50], "max_depth": [10]}
    #
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring="roc_auc", cv=2, verbose=2)

    # nn = MLPClassifier(max_iter=500, random_state=42)
    #
    # # 定义超参数搜索空间
    # param_grid = {
    #     "hidden_layer_sizes": [(64, 32)],  # 隐藏层结构
    #     "activation": ["relu", "tanh"],  # 激活函数
    #     "solver": ["adam"],  # 优化器
    #     "alpha": [0.01]  # L2 正则化强度
    # }
    #
    # # 进行超参数搜索
    # grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, scoring="roc_auc", cv=2, verbose=2)

    # lr = LogisticRegression()
    # param_grid = {"C": [0.1, 1, 10]}
    #
    # grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, scoring="roc_auc", cv=2, verbose=2)


    # svc = SVC(probability=True)
    # param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    #
    # grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring="roc_auc", cv=2, verbose=2)


    # mlp = MLPClassifier()
    # param_grid = {"hidden_layer_sizes": [(20,)], "alpha": [0.001]}
    #
    # grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring="roc_auc", cv=2, verbose=2)


    # knn = KNeighborsClassifier()
    # param_grid = {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]}
    #
    # grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring="roc_auc", cv=2, verbose=2)

    # Train the model with grid search
    grid_search.fit(X_train, y_train.values.ravel())
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

    # TODO config save or not by args
    # save train test set to csv
    # pd.DataFrame(y_pred).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-{args.division}-y_pred.csv", index=False)

    # # 用于RF的参数重要性分析
    # rf = RandomForestClassifier(n_estimators=50, max_depth=10)
    # rf.fit(X_train, y_train.values.ravel())
    # feature_importance = rf.feature_importances_
    # # 将特征名称和重要性存入 DataFrame 进行排序
    # feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]  # 自动生成特征名
    # importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    # importance_df = importance_df.sort_values(by="Importance", ascending=False)
    # # 打印特征重要性
    # print("特征重要性排名：")
    # print(importance_df)
    # # 可视化特征重要性
    # plt.figure(figsize=(10, 6))
    # plt.barh(importance_df["Feature"], importance_df["Importance"], color="royalblue")
    # plt.xlabel("Importance Score")
    # plt.ylabel("Feature")
    # plt.title("Feature Importance in Random Forest")
    # plt.gca().invert_yaxis()  # 最高重要性的特征放在顶部
    # plt.show()

    # TODO: work mode data input to model and predict


if __name__ == '__main__':
    main()
