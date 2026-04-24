from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Data" / "telecom_churn.csv"

FEATURE_COLS = [
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
TARGET_COL = "Churn"


class ChurnInput(BaseModel):
    AccountWeeks: float
    ContractRenewal: int
    DataPlan: int
    DataUsage: float
    CustServCalls: float
    DayMins: float
    DayCalls: float
    MonthlyCharge: float
    OverageFee: float
    RoamMins: float


def build_models(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    logistic_model.fit(X_train, y_train)

    tree_model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
        max_depth=5,
        min_samples_leaf=20,
    )
    tree_model.fit(X_train, y_train)

    return logistic_model, tree_model, X, y


app = FastAPI(title="Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/charts", StaticFiles(directory=BASE_DIR), name="charts")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontro el dataset en: {DATA_PATH}")

_df = pd.read_csv(DATA_PATH)
logistic_model, tree_model, full_X, full_y = build_models(_df)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "rows": int(_df.shape[0]), "columns": int(_df.shape[1])}


@app.get("/eda-images")
def eda_images() -> dict:
    files = sorted(BASE_DIR.glob("grafico_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "count": len(files),
        "images": [f"/charts/{f.name}" for f in files],
    }


@app.post("/predict")
def predict(payload: ChurnInput) -> dict:
    row = pd.DataFrame([payload.model_dump()])[FEATURE_COLS]

    logistic_pred = int(logistic_model.predict(row)[0])
    logistic_prob = float(logistic_model.predict_proba(row)[0][1])

    tree_pred = int(tree_model.predict(row)[0])
    tree_prob = float(tree_model.predict_proba(row)[0][1])

    return {
        "logistic": {"prediction": logistic_pred, "churn_probability": round(logistic_prob, 4)},
        "tree_entropy": {"prediction": tree_pred, "churn_probability": round(tree_prob, 4)},
    }


@app.get("/predictions")
def predictions(
    model: Literal["logistic", "tree"] = Query(default="tree"),
    limit: int = Query(default=200, ge=1, le=3333),
) -> dict:
    selected_model = logistic_model if model == "logistic" else tree_model

    y_pred = selected_model.predict(full_X)
    y_prob = selected_model.predict_proba(full_X)[:, 1]

    result_df = _df.copy()
    result_df["predicted_churn"] = y_pred.astype(int)
    result_df["predicted_probability"] = y_prob.round(4)

    response_rows = result_df.head(limit).to_dict(orient="records")

    return {
        "model": model,
        "total_rows": int(result_df.shape[0]),
        "returned_rows": int(len(response_rows)),
        "predicted_churn_count": int(result_df["predicted_churn"].sum()),
        "rows": response_rows,
    }
