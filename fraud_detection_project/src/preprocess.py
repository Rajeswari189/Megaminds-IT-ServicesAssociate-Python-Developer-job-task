# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("Columns in dataset:", df.columns.tolist())

    scaler = StandardScaler()

    # Handle Amount column safely
    if 'Amount' in df.columns:
        df['Amount'] = scaler.fit_transform(df[['Amount']])
    elif 'amount' in df.columns:
        df['amount'] = scaler.fit_transform(df[['amount']])
    else:
        print("⚠️ Warning: No Amount column found. Skipping amount scaling.")

    # Drop Time column if exists
    if 'Time' in df.columns:
        df.drop(columns=['Time'], inplace=True)
    elif 'time' in df.columns:
        df.drop(columns=['time'], inplace=True)

    return df
