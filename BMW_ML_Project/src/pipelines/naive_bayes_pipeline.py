from typing import Dict, Any, Optional, List
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

from src.preprocess import build_preprocessor
from src.metrics import compute_classification_metrics, format_metrics

def build_nb_pipeline(
    scaled_cols: List[str],
    categorical_cols: List[str],
    var_smoothing: float = 1e-8
) -> Pipeline:
    preprocessor = build_preprocessor(scaled_cols, categorical_cols)
    pipe = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', GaussianNB(var_smoothing=var_smoothing))
    ])
    return pipe

def train_pipeline(pipe: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe.fit(x_train, y_train)
    return pipe

def evaluate_pipeline(pipe: Pipeline, x: pd.DataFrame, y: pd.Series, average: str="weighted") -> Dict[str, float]:
    preds = pipe.predict(x)
    return compute_classification_metrics(y, preds, average=average)

def train_eval_save(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    scaled_cols: List[str],
    categorical_cols: List[str],
    var_smoothing: float = 1e-8,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    pipe = build_nb_pipeline(scaled_cols, categorical_cols, var_smoothing)
    train_pipeline(pipe, x_train, y_train)
    train_metrics = evaluate_pipeline(pipe, x_train, y_train)
    test_metrics  = evaluate_pipeline(pipe, x_test,  y_test)

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipe, model_path)

    return {"pipeline": pipe, "train_metrics": train_metrics, "test_metrics": test_metrics, "model_path": model_path}

def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)

def print_summary(train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> None:
    print("\nnaive bayes (pipeline) results:")
    print(format_metrics("train", train_metrics))
    print(format_metrics("test ", test_metrics))
