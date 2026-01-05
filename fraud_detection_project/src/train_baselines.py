# train LR, RF, XGBoost baselines
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def train_and_eval_baselines(X_train, y_train, X_val, y_val):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)

        results[name] = auc
        print(f"{name} AUC: {auc:.4f}")

    return results
