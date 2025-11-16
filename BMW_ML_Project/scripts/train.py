import argparse, os, pandas as pd, yaml, mlflow, mlflow.sklearn
from src.split_data import split_dataset
from src.pipelines.naive_bayes_pipeline import train_eval_save, print_summary
def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train GaussianNB pipeline with MLflow")
    parser.add_argument("--data", required=True, help="Path to CSV (clean) data file")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--out", default="artifacts/nb_pipeline.joblib", help="Where to save the trained pipeline")
    parser.add_argument("--experiment", default="sales_classification_nb", help="MLflow experiment name")
    parser.add_argument("--tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"), help="MLflow tracking URI")
    parser.add_argument("--register_as", default="", help="Optional MLflow model name to register")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    cfg = load_config(args.config)
    df = pd.read_csv(args.data)


    x_train, x_test, y_train, y_test = split_dataset(
        df=df,
        target_col=cfg["target_col"],
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 42),
    )

    with mlflow.start_run() as run:
        out = train_eval_save(
            x_train=x_train, y_train=y_train,
            x_test=x_test,   y_test=y_test,
            scaled_cols=cfg["features"]["scaled_cols"],
            categorical_cols=cfg["features"]["categorical_cols"],
            var_smoothing=float(cfg["model"].get("var_smoothing", 1e-8)),
            model_path=args.out
        )
        print_summary(out["train_metrics"], out["test_metrics"])
        print(f"\nSaved pipeline -> {args.out}")

        mlflow.log_param("model_type", "GaussianNB")
        mlflow.log_param("var_smoothing", cfg["model"].get("var_smoothing", 1e-8))
        mlflow.log_param("scaled_cols", ",".join(cfg["features"]["scaled_cols"]))
        mlflow.log_param("categorical_cols", ",".join(cfg["features"]["categorical_cols"]))
        mlflow.log_param("target_col", cfg["target_col"])
        mlflow.log_param("test_size", cfg.get("test_size", 0.2))
        mlflow.log_param("random_state", cfg.get("random_state", 42))

        for k, v in out["train_metrics"].items():
            mlflow.log_metric(f"train_{k}", float(v))
        for k, v in out["test_metrics"].items():
            mlflow.log_metric(f"test_{k}", float(v))

        mlflow.sklearn.log_model(out["pipeline"], "model")
        mlflow.log_artifact(args.config)

        if args.register_as:
            try:
                mv = mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/model",
                    name=args.register_as
                )
                print(f"Registered model as {args.register_as} -> version {mv.version}")
            except Exception as e:
                print(f"Model registry not available or failed: {e}")

if __name__ == "__main__":
    main()
