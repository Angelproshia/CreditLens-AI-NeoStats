from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


def risk_band(probability: float) -> str:
    return "Low" if probability < 0.10 else "Medium" if probability < 0.25 else "High"


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical),
    ])
    classifier = LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(df: pd.DataFrame, features: list[str], output_path: str | Path):
    X, y = df[features].copy(), df["TARGET"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "test_rows": len(y_test),
        "default_rate": float(y.mean()),
    }

    # A shallow surrogate tree turns model behaviour into readable policy hints.
    transformed = pipeline.named_steps["preprocessor"].transform(X_train)
    surrogate = DecisionTreeClassifier(max_depth=3, min_samples_leaf=250, random_state=42)
    surrogate.fit(transformed, pipeline.predict(X_train))
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    rules = export_text(surrogate, feature_names=list(feature_names), decimals=2)

    bundle = {"pipeline": pipeline, "features": features, "metrics": metrics,
              "rules": rules, "training_sample": X_train.head(250)}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return bundle


def load_bundle(path):
    return joblib.load(path)


def predict_one(bundle, applicant: dict):
    row = pd.DataFrame([{f: applicant.get(f) for f in bundle["features"]}])
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return {"default_probability": probability, "risk_score": round(probability * 1000),
            "risk_band": risk_band(probability), "row": row}


def explain_linear_prediction(bundle, row: pd.DataFrame, top_n=8):
    import shap
    prep = bundle["pipeline"].named_steps["preprocessor"]
    clf = bundle["pipeline"].named_steps["classifier"]
    values = prep.transform(row)
    background = prep.transform(bundle["training_sample"])
    if hasattr(values, "toarray"):
        values = values.toarray()
    if hasattr(background, "toarray"):
        background = background.toarray()
    names = prep.get_feature_names_out()
    explainer = shap.LinearExplainer(clf, background)
    contributions = np.asarray(explainer(values).values)[0]
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return pd.DataFrame({"feature": names[order], "contribution": contributions[order]}).sort_values("contribution")
