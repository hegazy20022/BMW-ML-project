    .PHONY: install train api ui test lint fmt mlflow

    install:
	pip install -r requirements.txt

    train:
	python scripts/train.py --data data/clean.csv --config configs/config.yaml --out artifacts/nb_pipeline.joblib

    api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000

    ui:
	streamlit run ui/streamlit_app.py

    test:
	pytest -q

    lint:
	ruff check . && mypy .

    fmt:
	black . && ruff check . --fix

    mlflow:
	mlflow ui --backend-store-uri ${MLFLOW_BACKEND_URI:-file:./mlruns} --host 0.0.0.0 --port 5000
