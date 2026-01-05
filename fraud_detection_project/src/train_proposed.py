# trains HFS-XGBoost-AE pipeline
import numpy as np
from xgboost import XGBClassifier
from src.autoencoder import build_autoencoder
from sklearn.metrics import roc_auc_score

def train_proposed_model(X_train, y_train, X_val, y_val, selected_features):
    # Train autoencoder on non-fraud samples
    X_train_nf = X_train[y_train == 0][selected_features]

    ae = build_autoencoder(X_train_nf.shape[1])
    ae.fit(
        X_train_nf,
        X_train_nf,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        verbose=0
    )

    def anomaly_score(model, X):
        recon = model.predict(X)
        return ((X - recon) ** 2).mean(axis=1)

    train_scores = anomaly_score(ae, X_train[selected_features])
    val_scores = anomaly_score(ae, X_val[selected_features])

    X_train_fused = np.column_stack([X_train[selected_features], train_scores])
    X_val_fused = np.column_stack([X_val[selected_features], val_scores])

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_fused, y_train)

    probs = model.predict_proba(X_val_fused)[:, 1]
    auc = roc_auc_score(y_val, probs)

    print(f"Proposed Model AUC: {auc:.4f}")

    return model, ae
