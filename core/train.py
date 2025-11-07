import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def train_model(
        model_obj: LinearRegression | RandomForestRegressor | SVR,
        independent_vars: pd.DataFrame,
        dependent_var: pd.Series,
) -> LinearRegression | RandomForestRegressor | SVR:

    model_obj.fit(independent_vars, dependent_var)
    print(f"Trained model: {model_obj.__class__.__name__} \nScore: {model_obj.score(independent_vars, dependent_var)}\n\n")
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
        "svr": SVR(kernel="linear", C=1.0),
        "radom_forest": RandomForestRegressor()
    }

    for model in models:
        models[model] = train_model(models[model], X_train, y_train)


