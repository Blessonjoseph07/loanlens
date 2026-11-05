"""Model shootout: five classifiers on the same split as train.py.

Usage: python compare_models.py loan_approval_dataset.csv
Output: comparison.json (read by the Model performance tab)

Logistic Regression and Random Forest use exactly the settings in train.py,
so their numbers match metrics.json. Three gradient-boosting libraries are
added: scikit-learn's HistGradientBoosting, XGBoost and LightGBM. Every
model gets the same preprocessing pipeline, the same stratified 80/20 split
(seed 42) and the same 5-fold CV on the training part. Besides quality, the
practical costs are measured on this machine: training time, latency for a
single applicant and the size of the saved model.

This script does NOT replace model.joblib - the app keeps the Random Forest,
because SHAP's TreeExplainer and the rest of the app are built around it.
"""
import io
import json
import sys
import time

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from loanlens.data import CATEGORICAL, load_and_clean

SEED = 42


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, random_state=SEED,
                                 n_jobs=-1, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, n_jobs=-1,
                                   verbose=-1),
    }


def main(path: str) -> None:
    df = load_and_clean(path)
    y = (df.pop("loan_status") == "Approved").astype(int)
    X = df
    num = [c for c in X.columns if c not in CATEGORICAL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    one = X_te.iloc[[0]]

    results = {}
    for name, clf in models().items():
        pre = ColumnTransformer([("num", StandardScaler(), num),
                                 ("cat", OneHotEncoder(drop="if_binary"), CATEGORICAL)])
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1")
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        prob = pipe.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        pipe.predict_proba(one)                                   # warm-up
        t0 = time.perf_counter()
        for _ in range(50):
            pipe.predict_proba(one)
        latency_ms = (time.perf_counter() - t0) / 50 * 1000
        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        fpr, tpr, _ = roc_curve(y_te, prob)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 80)).astype(int))
        results[name] = {
            "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, prob),
            "cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std()),
            "errors": int((pred != y_te.to_numpy()).sum()),
            "train_seconds": round(train_s, 3), "latency_ms": round(latency_ms, 2),
            "size_kb": round(buf.getbuffer().nbytes / 1024, 1),
            "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        }
        r = results[name]
        print(f"{name:22s} acc {r['accuracy']:.3f}  f1 {r['f1']:.3f}  auc {r['roc_auc']:.4f}  "
              f"errors {r['errors']:3d}  train {r['train_seconds']:.2f}s  "
              f"latency {r['latency_ms']:.1f} ms  size {r['size_kb']:.0f} KB")

    json.dump({"test_size": int(len(y_te)), "models": results}, open("comparison.json", "w"),
              indent=2, default=float)
    print("Saved comparison.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "loan_approval_dataset.csv")
