import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def train_val_test_split(df, target='Class', test_size=0.2, val_size=0.1):
    X = df.drop(columns=[target])
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, stratify=y_train_full, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
