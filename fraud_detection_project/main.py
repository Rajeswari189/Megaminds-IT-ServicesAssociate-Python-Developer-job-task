from src.utils import load_data, train_val_test_split
from src.preprocess import preprocess_data
from src.feature_selection import hybrid_feature_selection
from src.train_baselines import train_and_eval_baselines
from src.train_proposed import train_proposed_model

def main():
    print("Starting Fraud Detection Pipeline...")

    df = load_data("data/creditcard.csv")
    df = preprocess_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    selected_features = hybrid_feature_selection(X_train, y_train)
    print("Selected Features:", selected_features)

    print("\nTraining baseline models...")
    train_and_eval_baselines(
        X_train[selected_features], y_train,
        X_val[selected_features], y_val
    )

    print("\nTraining proposed model...")
    train_proposed_model(
        X_train[selected_features], y_train,
        X_val[selected_features], y_val,
        selected_features
    )

    print("\nPipeline executed successfully! ")

if __name__ == "__main__":
    main()
