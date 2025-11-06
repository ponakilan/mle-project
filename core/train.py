import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_CSV_PATH = "data/combined_dataset.csv"
TARGET_COL = "LN_IC50"
RANDOM_SEED = 42

train_df = pd.read_csv(TRAIN_CSV_PATH)
X, y = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


