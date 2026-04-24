import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("telecom_churn.csv")
target_col = "Churn"
feature_cols = [
    "AccountWeeks",
    "ContractRenewal",
    "DataPlan",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins",
]

X = df[feature_cols]
y = df[target_col]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 80)
print("MODELADO PREDICTIVO - CHURN")
print("=" * 80)
print(f"Train: {X_train.shape[0]} filas ({(X_train.shape[0] / len(df)) * 100:.0f}%)")
print(f"Validacion: {X_valid.shape[0]} filas ({(X_valid.shape[0] / len(df)) * 100:.0f}%)")

models = {
    "Regresion Logistica": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    ),
    "Arbol de Decision (Entropia)": DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
        max_depth=5,
        min_samples_leaf=20,
    ),
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    prec = precision_score(y_valid, y_pred, zero_division=0)
    rec = recall_score(y_valid, y_pred, zero_division=0)
    f1 = f1_score(y_valid, y_pred, zero_division=0)

    results.append(
        {
            "modelo": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    )

    cm = confusion_matrix(y_valid, y_pred)

    print("\n" + "-" * 80)
    print(f"Modelo: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Matriz de confusion:")
    print(cm)

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print("\n" + "=" * 80)
print("Comparativa de modelos")
print("=" * 80)
print(results_df.round(4))
