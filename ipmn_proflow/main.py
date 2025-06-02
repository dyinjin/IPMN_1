from imports import *

y_pred = None

def main():
    """
    Main function to execute the program logic.
    """
    global y_pred
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
        # Thought the time span was too short(1day), NOT SUITABLE as a training set
        data_set = UnitDataLoader.csvloader_specified(config, config.IBM_CSV)
        data_set = UnitDataLoader.datauniter_ibm(config, data_set)
        train_set, test_set = train_test_split(data_set, test_size=0.3, random_state=config.RANDOM_SEED)
        train_set = train_set.sort_values(by='Date')
        test_set = test_set.sort_values(by='Date')
        # split_index = int(len(data_set) * 0.7)
        # train_set, test_set = data_set.iloc[:split_index], data_set.iloc[split_index:]
    elif args.dataset == config.DATASET_MODES['specific_train_specific_test']:
        train_set = UnitDataLoader.csvloader_specified(config, "2022-11.csv")
        train_set = UnitDataLoader.datauniter_saml(config, train_set)
        # need different datauniter, the column names of the dataset need be consistent
        test_set = UnitDataLoader.csvloader_specified(config, "2023-06.csv")
        test_set = UnitDataLoader.datauniter_saml(config, test_set)

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

    if config.SAVE_TRAIN_TEST == 0:
        pass
    elif config.SAVE_TRAIN_TEST == 1:
        # save train/test X/y to csv
        X_train.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_train.csv", index=False)
        X_test.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test.csv", index=False)
        y_train.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_train.csv", index=False)
        y_test.to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_test.csv", index=False)
    elif config.SAVE_TRAIN_TEST == 2:
        # save train/test set to csv
        pd.concat([X_train, y_train], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_train_with_y.csv", index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test_with_y.csv", index=False)

    # TODO: Implement model choice functionality
    # Process numerical and categorical features for classification models
    numerical_features = X_train.select_dtypes(exclude="object").columns
    categorical_features = X_train.select_dtypes(include="object").columns

    # "account" always categorical
    account_columns = [col for col in numerical_features if "account" in col]
    numerical_features = numerical_features.difference(account_columns)
    categorical_features = categorical_features.union(account_columns)

    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    transformer = ColumnTransformer(transformers=[
        # Encode categorical features using OrdinalEncoder
        ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        # Scale numerical features using RobustScaler
        ("RobustScaler", RobustScaler(), numerical_features)
    ], remainder="passthrough")

    # TODO: ColumnTransformer前后合理性

    # Apply transformations to training and testing datasets
    columns_name = X_train.columns
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    # 保存 transformer
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.SAVE_TRAIN_TEST != 0:
        transformer_path = f"{config.DATAPATH}{current_time}_{config.SAVE_TRANS}"
        joblib.dump(transformer, transformer_path)
        print(f"Transformer saved to {transformer_path}")

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
    # param_grid = {
    #     "hidden_layer_sizes": [(64, 32)],
    #     "activation": ["relu", "tanh"],
    #     "solver": ["adam"],
    #     "alpha": [0.01]
    # }
    #
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

    # 存储模型
    if config.SAVE_TRAIN_TEST != 0:
        model_path = f"{config.DATAPATH}{current_time}_{config.SAVE_MODEL}"
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

    # Evaluate the model using ROC-AUC score
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probabilities)
    print("Test AUC: ", test_auc)

    if config.SAVE_TRAIN_TEST == 0:
        pass
    elif config.SAVE_TRAIN_TEST == 1:
        pd.DataFrame(test_probabilities).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_prob.csv", index=False)
    elif config.SAVE_TRAIN_TEST == 2:
        test_probabilities = pd.Series(test_probabilities, name="predict_fraud_probability")
        pd.concat([pd.DataFrame(X_test, columns=columns_name), y_test, test_probabilities], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test_with_prob.csv", index=False)

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, test_probabilities)

    # 绘制 ROC 曲线
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    # 判断是否使用预设 TPR、鼠标点击值或保留概率数值
    if config.TPR_SET == 0:
        # 使用预设 TPR
        desired_tpr = config.TPR
        closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]
        y_pred = (test_probabilities >= closest_threshold).astype(int)
        print(f"Using preset TPR: {desired_tpr}, Closest Threshold: {closest_threshold}")

    elif config.TPR_SET == 1:
        def onclick(event):
            global y_pred
            if event.xdata is not None and event.ydata is not None:
                desired_tpr = event.ydata  # 用户选择的 TPR 值
                closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]

                # 根据阈值计算预测标签
                y_pred = (test_probabilities >= closest_threshold).astype(int)
                print(f"Selected TPR: {desired_tpr}, Closest Threshold: {closest_threshold}")

        # 绑定鼠标点击事件
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

    if config.SAVE_TRAIN_TEST == 0:
        pass
    elif config.SAVE_TRAIN_TEST == 1:
        pd.DataFrame(y_pred).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-y_pred.csv", index=False)
    elif config.SAVE_TRAIN_TEST == 2:
        y_pred = pd.Series(y_pred, name="predict_fraud")
        pd.concat([pd.DataFrame(X_test, columns=columns_name), y_test, y_pred], axis=1).to_csv(f"{config.DATAPATH}{args.dataset}-{args.param}-X_test_with_pred.csv", index=False)

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

    # TODO: show tree, importance, graph


if __name__ == '__main__':
    main()
