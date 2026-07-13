"""
Physics-Guided Surrogate Model Service
========================================
FastAPI microservice that learns to approximate the Monte Carlo
physics-based DCR (Demand-Capacity Ratio) computation for structural
reliability analysis, replacing the expensive per-sample formula loop
with a fast trained model (XGBoost or Gaussian Process Regression).

Endpoints:
  POST /train    - train a surrogate model on Monte Carlo samples
  POST /predict  - predict DCR / failure for new inputs using a trained model
  GET  /health   - health check
  GET  /models   - list currently trained/loaded models
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

import xgboost as xgb

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "profile_height_m",
    "profile_width_m",
    "cross_section_area_m2",
    "f_ck",
    "f_yk",
    "dead_load",
    "live_load",
    "corrosion_rate_pct_per_year",
    "timeStep",
]
CATEGORICAL_FEATURES = ["ifc_class"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = ["dcr_healthy", "dcr_damaged"]

VALID_MODEL_TYPES = ("xgboost", "gpr")

# In-memory registry: model_type -> {"preprocessor", "dcr_healthy", "dcr_damaged", "meta"}
MODELS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="Structural Reliability Surrogate Model Service",
    description=(
        "Physics-guided surrogate model for time-dependent structural "
        "reliability analysis. Learns the input-output mapping produced "
        "by Monte Carlo physics-based simulation and serves fast predictions."
    ),
    version="1.0.0",
)


# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------

class TrainRecord(BaseModel):
    ifc_class: str
    profile_height_m: float
    profile_width_m: float
    cross_section_area_m2: float
    f_ck: Optional[float] = None
    f_yk: float
    dead_load: float
    live_load: float
    corrosion_rate_pct_per_year: float
    timeStep: float
    dcr_healthy: float
    dcr_damaged: float


class PredictRecord(BaseModel):
    ifc_class: str
    profile_height_m: float
    profile_width_m: float
    cross_section_area_m2: float
    f_ck: Optional[float] = None
    f_yk: float
    dead_load: float
    live_load: float
    corrosion_rate_pct_per_year: float
    timeStep: float


class TrainRequest(BaseModel):
    model_type: str = Field(default="xgboost", description="'xgboost' or 'gpr'")
    records: List[TrainRecord]
    max_samples_gpr: int = Field(
        default=3000,
        description="GPR training cost scales O(n^3); large datasets are auto-subsampled.",
    )


class PredictRequest(BaseModel):
    model_type: str = Field(default="xgboost", description="'xgboost' or 'gpr'")
    records: List[PredictRecord]


class TrainResponse(BaseModel):
    model_type: str
    n_samples_available: int
    n_samples_used: int
    training_time_seconds: float
    r2_dcr_healthy: float
    r2_dcr_damaged: float
    mae_dcr_healthy: float
    mae_dcr_damaged: float
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None


class PredictionResult(BaseModel):
    dcr_healthy: float
    dcr_damaged: float
    dcr_healthy_std: Optional[float] = None
    dcr_damaged_std: Optional[float] = None
    failure_healthy: bool
    failure_damaged: bool


class PredictResponse(BaseModel):
    model_type: str
    n_predictions: int
    prediction_time_seconds: float
    predictions: List[PredictionResult]


# --------------------------------------------------------------------------
# Model persistence helpers
# --------------------------------------------------------------------------

def _paths(model_type: str):
    return {
        "preprocessor": MODELS_DIR / f"{model_type}_preprocessor.pkl",
        "dcr_healthy": MODELS_DIR / f"{model_type}_dcr_healthy.pkl",
        "dcr_damaged": MODELS_DIR / f"{model_type}_dcr_damaged.pkl",
        "meta": MODELS_DIR / f"{model_type}_meta.json",
    }


def save_model_to_disk(model_type: str, preprocessor, model_healthy, model_damaged, meta: dict):
    p = _paths(model_type)
    joblib.dump(preprocessor, p["preprocessor"])
    joblib.dump(model_healthy, p["dcr_healthy"])
    joblib.dump(model_damaged, p["dcr_damaged"])
    with open(p["meta"], "w") as f:
        json.dump(meta, f)


def load_model_from_disk(model_type: str) -> Optional[dict]:
    p = _paths(model_type)
    if not all(path.exists() for path in p.values()):
        return None
    return {
        "preprocessor": joblib.load(p["preprocessor"]),
        "dcr_healthy": joblib.load(p["dcr_healthy"]),
        "dcr_damaged": joblib.load(p["dcr_damaged"]),
        "meta": json.load(open(p["meta"])),
    }


@app.on_event("startup")
def load_existing_models():
    for model_type in VALID_MODEL_TYPES:
        loaded = load_model_from_disk(model_type)
        if loaded:
            MODELS[model_type] = loaded
            print(f"Loaded existing '{model_type}' model from disk.")


def get_model(model_type: str) -> dict:
    if model_type not in VALID_MODEL_TYPES:
        raise HTTPException(400, f"model_type must be one of {VALID_MODEL_TYPES}")
    if model_type in MODELS:
        return MODELS[model_type]
    loaded = load_model_from_disk(model_type)
    if loaded:
        MODELS[model_type] = loaded
        return loaded
    raise HTTPException(
        404,
        f"No trained '{model_type}' model found. Call POST /train first "
        f"with model_type='{model_type}'.",
    )


# --------------------------------------------------------------------------
# Preprocessing / model builders
# --------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_regressor(model_type: str):
    if model_type == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    if model_type == "gpr":
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e3)
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
        return GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform(
            pd.DataFrame(columns=ALL_FEATURES)
        ).shape[1])]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}


@app.get("/models")
def list_models():
    return {
        model_type: MODELS[model_type]["meta"]
        for model_type in MODELS
    }


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    if req.model_type not in VALID_MODEL_TYPES:
        raise HTTPException(400, f"model_type must be one of {VALID_MODEL_TYPES}")
    if len(req.records) < 10:
        raise HTTPException(400, "Need at least 10 training records.")

    df = pd.DataFrame([r.dict() for r in req.records])
    n_available = len(df)

    if req.model_type == "gpr" and n_available > req.max_samples_gpr:
        df = df.sample(n=req.max_samples_gpr, random_state=42).reset_index(drop=True)

    n_used = len(df)

    X = df[ALL_FEATURES]
    y_healthy = df["dcr_healthy"].values
    y_damaged = df["dcr_damaged"].values

    t0 = time.perf_counter()

    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X)

    model_healthy = build_regressor(req.model_type)
    model_healthy.fit(Xt, y_healthy)

    model_damaged = build_regressor(req.model_type)
    model_damaged.fit(Xt, y_damaged)

    training_time = time.perf_counter() - t0

    pred_healthy = model_healthy.predict(Xt)
    pred_damaged = model_damaged.predict(Xt)

    r2_healthy = float(r2_score(y_healthy, pred_healthy))
    r2_damaged = float(r2_score(y_damaged, pred_damaged))
    mae_healthy = float(mean_absolute_error(y_healthy, pred_healthy))
    mae_damaged = float(mean_absolute_error(y_damaged, pred_damaged))

    feature_importance = None
    if req.model_type == "xgboost":
        feat_names = get_feature_names(preprocessor)
        feature_importance = {
            "dcr_healthy": dict(zip(feat_names, [float(v) for v in model_healthy.feature_importances_])),
            "dcr_damaged": dict(zip(feat_names, [float(v) for v in model_damaged.feature_importances_])),
        }

    meta = {
        "model_type": req.model_type,
        "n_samples_available": n_available,
        "n_samples_used": n_used,
        "training_time_seconds": training_time,
        "r2_dcr_healthy": r2_healthy,
        "r2_dcr_damaged": r2_damaged,
        "mae_dcr_healthy": mae_healthy,
        "mae_dcr_damaged": mae_damaged,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    save_model_to_disk(req.model_type, preprocessor, model_healthy, model_damaged, meta)
    MODELS[req.model_type] = {
        "preprocessor": preprocessor,
        "dcr_healthy": model_healthy,
        "dcr_damaged": model_damaged,
        "meta": meta,
    }

    return TrainResponse(
        model_type=req.model_type,
        n_samples_available=n_available,
        n_samples_used=n_used,
        training_time_seconds=training_time,
        r2_dcr_healthy=r2_healthy,
        r2_dcr_damaged=r2_damaged,
        mae_dcr_healthy=mae_healthy,
        mae_dcr_damaged=mae_damaged,
        feature_importance=feature_importance,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(400, "records list is empty.")

    model = get_model(req.model_type)
    preprocessor = model["preprocessor"]
    model_healthy = model["dcr_healthy"]
    model_damaged = model["dcr_damaged"]

    df = pd.DataFrame([r.dict() for r in req.records])
    X = df[ALL_FEATURES]

    t0 = time.perf_counter()
    Xt = preprocessor.transform(X)

    if req.model_type == "gpr":
        mean_h, std_h = model_healthy.predict(Xt, return_std=True)
        mean_d, std_d = model_damaged.predict(Xt, return_std=True)
    else:
        mean_h = model_healthy.predict(Xt)
        std_h = [None] * len(mean_h)
        mean_d = model_damaged.predict(Xt)
        std_d = [None] * len(mean_d)

    prediction_time = time.perf_counter() - t0

    predictions = []
    for i in range(len(df)):
        dcr_h = max(0.0, float(mean_h[i]))
        dcr_d = max(0.0, float(mean_d[i]))
        predictions.append(PredictionResult(
            dcr_healthy=dcr_h,
            dcr_damaged=dcr_d,
            dcr_healthy_std=float(std_h[i]) if std_h[i] is not None else None,
            dcr_damaged_std=float(std_d[i]) if std_d[i] is not None else None,
            failure_healthy=dcr_h >= 1.0,
            failure_damaged=dcr_d >= 1.0,
        ))

    return PredictResponse(
        model_type=req.model_type,
        n_predictions=len(predictions),
        prediction_time_seconds=prediction_time,
        predictions=predictions,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
