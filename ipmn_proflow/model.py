from imports import *


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


def search_best_save(config, trained_grid_search_model, model_id):
    # TODO: find if any other method for best_estimator
    print("Best Parameters: ", trained_grid_search_model.best_params_)
    best_model = trained_grid_search_model.best_estimator_

    # 存储模型
    if config.SAVE_LEVEL != -1:
        model_path = f"{config.DATAPATH}{model_id}_{config.SAVE_MODEL}"
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

    return best_model


def test_model(best_model, X_test):
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    return test_probabilities
