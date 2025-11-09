import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def train_model(
        model_obj: LinearRegression | RandomForestRegressor | SVR,
        independent_vars: pd.DataFrame,
        dependent_var: pd.Series,
) -> LinearRegression | RandomForestRegressor | SVR:

    if model_obj.__class__.__name__ == "SVR":
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.001],
            'epsilon': [0.05, 0.1, 0.2, 0.5],
            'kernel': ['rbf']
        }

        grid_search = GridSearchCV(
            estimator=model_obj,
            param_grid=param_grid,
            scoring='r2',
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(independent_vars, dependent_var)
        print("Best Params:", grid_search.best_params_)
        print(f"Trained model: {model_obj.__class__.__name__} \nScore: {grid_search.best_score_}\n")
        model_obj = grid_search.best_estimator_
    else:
        model_obj.fit(independent_vars, dependent_var)
        print(f"Trained model: {model_obj.__class__.__name__} \nScore: {model_obj.score(independent_vars, dependent_var)}\n")
    # model_obj.fit(independent_vars, dependent_var)
    # print(f"Trained model: {model_obj.__class__.__name__} \nScore: {model_obj.score(independent_vars, dependent_var)}\n")
    return model_obj


if __name__ == "__main__":
    TRAIN_CSV_PATH = "data/preprocessed_dataset.csv"
    TARGET_COL = "LN_IC50"
    RANDOM_SEED = 42

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    X, y = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    models = {
        "linear_regression": LinearRegression(),
        "svr": SVR(kernel="rbf", C=1.0, epsilon=0.5, gamma=0.001),
        "random_forest": RandomForestRegressor(),
        "xgb": xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    for model in models:
        models[model] = train_model(models[model], X_train, y_train)

    with open("data/models.pkl", "wb") as f:
        pickle.dump(models, f)
