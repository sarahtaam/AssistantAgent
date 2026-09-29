"""
recouvrement/train.py — Trains the LightGBM "will this client pay" model.

Usage:
    python recouvrement/train.py
    python recouvrement/train.py --data data/credit_score_cleaned.csv --threshold 0.40

If no dataset is found at --data, a synthetic one is generated so the
project trains out of the box. Drop in the real public credit-score
dataset (same column schema) at that path for a faithful reproduction
of the original results.

Outputs:
    recouvrement/model_lgbm.pkl
    recouvrement/model_features.pkl
    reports/metrics.json
    reports/confusion_matrix.png
    reports/feature_importance.png
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

FEATURES = [
    'retard_moyen_jours', 'nb_retards', 'dette_impayee',
    'anciennete_jours', 'nb_credits', 'taux_interet',
    'nb_comptes', 'ratio_utilisation',
    'score_retard', 'ratio_dette_anciennete',
    'risque_global', 'charge_credit', 'complexite_financiere',
]


def _generate_synthetic_dataset(path: str, n: int = 4000, seed: int = 42) -> None:
    """Fallback so the pipeline is runnable without downloading a dataset.
    Mirrors the column schema of the public Kaggle credit-score dataset
    this project was originally built against."""
    rng = np.random.default_rng(seed)
    delay = rng.gamma(2, 8, n).clip(0, 90)
    n_delayed = rng.poisson(3, n)
    debt = rng.gamma(2, 400, n)
    history_age = rng.uniform(0, 400, n)
    n_loan = rng.integers(0, 8, n)
    interest = rng.uniform(4, 34, n)
    n_accounts = rng.integers(1, 10, n)
    utilization = rng.uniform(10, 50, n)

    risk = (delay / 90 + n_delayed / 10 + debt / 2000 - history_age / 500).clip(0, None)
    prob_bad = 1 / (1 + np.exp(-(risk * 3 - 2)))
    label = rng.binomial(1, prob_bad)
    credit_score = np.where(label == 1, rng.choice(["Standard", "Poor"], n), "Good")

    pd.DataFrame({
        "credit_score": credit_score,
        "delay_from_due_date": delay,
        "num_of_delayed_payment": n_delayed,
        "outstanding_debt": debt,
        "credit_history_age": history_age,
        "num_of_loan": n_loan,
        "interest_rate": interest,
        "num_bank_accounts": n_accounts,
        "credit_utilization_ratio": utilization,
    }).to_csv(path, index=False)
    print(f"No dataset found — generated a synthetic one at {path} ({n} rows).")


def _load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["credit_score"].map({"Good": 0, "Standard": 1, "Poor": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    mapped = pd.DataFrame({
        "retard_moyen_jours": df["delay_from_due_date"].clip(lower=0),
        "nb_retards": df["num_of_delayed_payment"],
        "dette_impayee": df["outstanding_debt"],
        "anciennete_jours": df["credit_history_age"].clip(lower=0),
        "nb_credits": df["num_of_loan"],
        "taux_interet": df["interest_rate"],
        "nb_comptes": df["num_bank_accounts"],
        "ratio_utilisation": df["credit_utilization_ratio"],
        "label": df["label"],
    }).dropna().reset_index(drop=True)

    # Outlier capping by median (mirrors the original cleaning step)
    for col, cap in [("retard_moyen_jours", mapped["retard_moyen_jours"].quantile(0.99)),
                      ("dette_impayee", mapped["dette_impayee"].quantile(0.99)),
                      ("ratio_utilisation", mapped["ratio_utilisation"].quantile(0.99))]:
        med = mapped.loc[mapped[col] <= cap, col].median()
        mapped.loc[mapped[col] > cap, col] = med

    mapped["score_retard"] = mapped["nb_retards"] * mapped["retard_moyen_jours"]
    mapped["ratio_dette_anciennete"] = mapped["dette_impayee"] / (mapped["anciennete_jours"] + 1)
    mapped["risque_global"] = mapped["nb_retards"] * mapped["taux_interet"] / (mapped["anciennete_jours"] + 1)
    mapped["charge_credit"] = mapped["dette_impayee"] / (mapped["nb_credits"] + 1)
    mapped["complexite_financiere"] = mapped["nb_comptes"] * mapped["nb_credits"]
    return mapped


def train(data_path: str, threshold: float, out_dir: str = "recouvrement", report_dir: str = "reports") -> dict:
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        _generate_synthetic_dataset(data_path)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df = _load_and_engineer(data_path)
    X, y = df[FEATURES].values, df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = dict(zip(classes.tolist(), weights))
    sw_train = np.array([cw[int(v)] for v in y_train])

    model = lgb.LGBMClassifier(
        n_estimators=700, max_depth=10, learning_rate=0.03, num_leaves=150,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.2, reg_lambda=0.2, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train, sample_weight=sw_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds) * 100, 2),
        "precision": round(precision_score(y_test, preds, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, preds, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, preds, zero_division=0) * 100, 2),
        "auc_roc": round(roc_auc_score(y_test, proba) * 100, 2),
        "threshold": threshold,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # 5-fold CV for a more honest generalization estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = []
    for tr_idx, val_idx in skf.split(X, y):
        ytr = y[tr_idx].astype(int)
        sw = np.array([cw[int(v)] for v in ytr])
        fm = lgb.LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=127,
                                 min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
        fm.fit(X[tr_idx], ytr, sample_weight=sw)
        cv_f1.append(f1_score(y[val_idx], fm.predict(X[val_idx]), zero_division=0))
    metrics["cv_f1_mean"] = round(float(np.mean(cv_f1)) * 100, 2)
    metrics["cv_f1_std"] = round(float(np.std(cv_f1)) * 100, 2)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Will pay", "Won't pay"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Will pay", "Won't pay"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # Feature importance plot
    importances = dict(zip(FEATURES, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(importances.keys())[::-1], list(importances.values())[::-1])
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "feature_importance.png"), dpi=120)
    plt.close(fig)

    joblib.dump(model, os.path.join(out_dir, "model_lgbm.pkl"))
    joblib.dump(FEATURES, os.path.join(out_dir, "model_features.pkl"))
    joblib.dump(threshold, os.path.join(out_dir, "model_threshold.pkl"))
    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/credit_score_cleaned.csv")
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()
    train(args.data, args.threshold)
