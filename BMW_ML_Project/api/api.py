import os
from typing import List, Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import mlflow.pyfunc
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import yaml


MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/nb_pipeline.joblib")
MODEL_URI = os.getenv("MODEL_URI", "")  
CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")

app = FastAPI(title="Sales Classification API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class Item(BaseModel):
    Mileage_KM: float
    Price_USD: float
    Sales_Volume: float
    Fuel_Type: str
    Color: str
    Region: str
    Model: str
    Transmission: str

class Items(BaseModel):
    records: List[Item]

model = None
model_is_mlflow = False
cfg: Dict[str, Any] | None = None

@app.on_event("startup")
def _load():
    global model, model_is_mlflow, cfg
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

    try:
        if MODEL_URI:
            model = mlflow.pyfunc.load_model(MODEL_URI)
            model_is_mlflow = True
        else:
            model = joblib.load(MODEL_PATH)
            model_is_mlflow = False
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/health", "version": "/version"}

@app.get("/health")
def health():
    return {"status": "ok", "model_type": "mlflow_pyfunc" if model_is_mlflow else "sklearn_pipeline"}

@app.get("/version")
def version():
    return {"api_version": app.version, "model_uri": MODEL_URI or MODEL_PATH}

def _coerce_input_df(df: pd.DataFrame) -> pd.DataFrame:
    if cfg is None:
        return df

    features = cfg.get("features", {})
    scaled_cols = features.get("scaled_cols", [])
    categorical_cols = features.get("categorical_cols", [])

    for c in scaled_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df

def _predict_df(df: pd.DataFrame):
    df = _coerce_input_df(df)
    if model_is_mlflow:
        preds = model.predict(df) 
    else:
        preds = model.predict(df) 
    return preds

@app.post("/predict")
@limiter.limit(RATE_LIMIT)
def predict(request: Request, item: Item) -> Dict[str, Any]:
    df = pd.DataFrame([item.model_dump()])
    try:
        pred = _predict_df(df)[0]
        return {"prediction": str(pred)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

@app.post("/batch_predict")
@limiter.limit(RATE_LIMIT)
def batch_predict(request: Request, items: Items) -> Dict[str, Any]:
    df = pd.DataFrame([it.model_dump() for it in items.records])
    try:
        preds = _predict_df(df).tolist()
        return {"predictions": [str(p) for p in preds]}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {e}")

@app.post("/predict_proba")
@limiter.limit(RATE_LIMIT)
def predict_proba(request: Request, items: Items) -> Dict[str, Any]:
    df = pd.DataFrame([it.model_dump() for it in items.records])
    df = _coerce_input_df(df)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)  
            classes = getattr(model, "classes_", list(range(proba.shape[1])))
            return {"classes": [str(c) for c in classes], "proba": proba.tolist()}
        else:
            if model_is_mlflow:
                raise HTTPException(status_code=400, detail="Model does not support predict_proba (mlflow)")
            raise HTTPException(status_code=400, detail="Model does not support predict_proba")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Predict_proba failed: {e}")


