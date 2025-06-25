from imports import *

# for roc click monitor needed. store predict output of test_set
y_pred = None
# for test and avoid retraining
best_model = None


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

    best_model = search_best_save(config, trained_grid_search_model, model_id)

    analysis_importance(best_model, columns_name)

    test_probabilities = test_model(best_model, X_test)

    save_predict_data2csv_float(config, test_probabilities, X_test, y_test, columns_name)

    analysis_performance(config, y_test, test_probabilities)

    save_predict_data2csv_bool(config, X_test, y_test, columns_name)


if __name__ == '__main__':
    main()
