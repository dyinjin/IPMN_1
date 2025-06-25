from imports import *


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
