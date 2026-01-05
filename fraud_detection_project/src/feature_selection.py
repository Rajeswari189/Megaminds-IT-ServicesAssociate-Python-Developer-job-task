import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def hybrid_feature_selection(X, y, k=10, corr_threshold=0.9):
    """
    Hybrid Feature Selection:
    - Mutual Information
    - Correlation filtering
    """
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns)

    # Select top-k features
    top_features = mi_series.sort_values(ascending=False).head(k).index.tolist()

    # Correlation filtering
    corr_matrix = X[top_features].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    drop_cols = [
        col for col in upper.columns if any(upper[col] > corr_threshold)
    ]

    selected_features = [f for f in top_features if f not in drop_cols]

    return selected_features
