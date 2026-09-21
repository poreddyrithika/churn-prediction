import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score


def run_validation():
    print("Starting Reproducibility Validation...")

    # 1. Load processed data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced"
    }

    # 2. Train two independent models with identical params/seed
    model_a = RandomForestClassifier(**params)
    model_a.fit(X_train, y_train)
    preds_a = model_a.predict(X_test)
    proba_a = model_a.predict_proba(X_test)[:, 1]

    model_b = RandomForestClassifier(**params)
    model_b.fit(X_train, y_train)
    preds_b = model_b.predict(X_test)
    proba_b = model_b.predict_proba(X_test)[:, 1]

    # 3. Compare predictions and metrics
    preds_match = np.array_equal(preds_a, preds_b)
    proba_match = np.allclose(proba_a, proba_b)

    f1_a = f1_score(y_test, preds_a)
    f1_b = f1_score(y_test, preds_b)
    auc_a = roc_auc_score(y_test, proba_a)
    auc_b = roc_auc_score(y_test, proba_b)

    print(f"Predictions identical : {preds_match}")
    print(f"Probabilities match   : {proba_match}")
    print(f"F1  (run A vs run B)  : {f1_a:.4f} vs {f1_b:.4f}")
    print(f"AUC (run A vs run B)  : {auc_a:.4f} vs {auc_b:.4f}")

    if preds_match and proba_match:
        print("\n[PASS] Model training is reproducible with fixed random_state.")
        sys.exit(0)
    else:
        print("\n[FAIL] Results differ between runs — check for non-deterministic steps.")
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
