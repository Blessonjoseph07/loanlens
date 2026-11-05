"""LoanLens - train and evaluate the loan approval models.

Usage: python train.py loan_approval_dataset.csv
Outputs: model.joblib, metrics.json, confusion_matrix.png, feature_importance.png

The cleaning step now lives in loanlens/data.py so the Streamlit app reuses
exactly the same code. The training logic and the numbers are unchanged.
"""
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loanlens.data import CATEGORICAL, load_and_clean

SEED = 42

# 1. Load and clean (column names and text values in this file have leading spaces)
df = load_and_clean(sys.argv[1] if len(sys.argv) > 1 else "loan_approval_dataset.csv")

print("Rows:", len(df), "| missing values:", int(df.isna().sum().sum()), "| duplicates:", int(df.duplicated().sum()))
print(df["loan_status"].value_counts())

y = (df.pop("loan_status") == "Approved").astype(int)
X = df
cat = CATEGORICAL
num = [c for c in X.columns if c not in cat]

# 2. Stratified 80/20 split; all fitting happens inside pipelines on the training set only
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(drop="if_binary"), cat)])
models = {
    "Majority class (baseline)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
}

results, cv = {}, StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1")
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    prob = pipe.predict_proba(X_te)[:, 1]
    results[name] = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, prob),
        "cv_f1_mean": cv_f1.mean(), "cv_f1_std": cv_f1.std(),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }
    models[name] = pipe
    print(f"\n{name}\n", classification_report(y_te, pred, target_names=["Rejected", "Approved"], zero_division=0))

rf = models["Random Forest"]

# 3. Feature importance
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=[n.split("__")[1] for n in names]).sort_values()
results["feature_importance"] = imp.sort_values(ascending=False).round(4).to_dict()

# 4. Edge case: same applicant, low CIBIL score vs high CIBIL score
base = X_te.iloc[[0]].copy()
for score in (350, 550, 750, 850):
    base["cibil_score"] = score
    results.setdefault("cibil_sweep", {})[score] = round(float(rf.predict_proba(base)[0, 1]), 3)

# 5. Misclassified test rows (failure cases)
pred = rf.predict(X_te)
wrong = X_te[pred != y_te].assign(actual=y_te[pred != y_te], prob_approved=rf.predict_proba(X_te[pred != y_te])[:, 1])
results["misclassified"] = wrong.round(3).to_dict(orient="records")

# 5b. Three real test-set applicants for the app's "Load sample applicant" menu:
#     a confident approval, a confident rejection, and the row whose approval
#     probability is closest to 0.5 (borderline). This only reads the test set;
#     it does not change any model or metric above.
test_prob = pd.Series(rf.predict_proba(X_te)[:, 1], index=X_te.index)
correct = test_prob[(test_prob >= 0.5).astype(int) == y_te]
picks = {
    "Approved applicant": correct[correct > 0.9].index[0],
    "Rejected applicant": correct[correct < 0.1].index[0],
    "Borderline applicant": (test_prob - 0.5).abs().idxmin(),
}
results["sample_applicants"] = {
    label: {**{k: (v.item() if hasattr(v, "item") else v) for k, v in X_te.loc[i].items()},
            "actual": "Approved" if y_te.loc[i] == 1 else "Rejected",
            "prob_approved": round(float(test_prob.loc[i]), 3)}
    for label, i in picks.items()
}

# 6. Save artefacts
joblib.dump(rf, "model.joblib")
json.dump(results, open("metrics.json", "w"), indent=2, default=float)

ConfusionMatrixDisplay(np.array(results["Random Forest"]["confusion_matrix"]),
                       display_labels=["Rejected", "Approved"]).plot(cmap="Blues", colorbar=False)
plt.title("Random Forest - test set confusion matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

imp.plot.barh(figsize=(7, 5), color="#4C72B0")
plt.title("Random Forest feature importance"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150); plt.close()
print("\nSaved model.joblib, metrics.json, confusion_matrix.png, feature_importance.png")
