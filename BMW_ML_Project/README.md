### Sales Classification ML Project 
A complete end-to-end machine learning project for predicting car sales categories using a Naive Bayes classifier.  
Includes: API (FastAPI), Streamlit UI, MLflow tracking, Docker support, visual analysis, and reproducible training pipeline.

###  Model Performance (Visualizations)

### Confusion Matrix
![confusion matrix](assets/confusion_matrix.png)

### ROC Curve
![ROC Curve](assets/roc_curve.png)

### classification report
![classification report](assets/classification_report.png)
### Quickstart (Colab / local)
```bash
pip install -r requirements.txt
python scripts/train.py --data data/clean.csv --config configs/config.yaml --out artifacts/nb_pipeline.joblib
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### MLflow
```bash
export MLFLOW_TRACKING_URI=file:./mlruns
mlflow ui --host 0.0.0.0 --port 5000
```

### Optional: Register model (requires MLflow registry backed by SQL DB)
```bash
python scripts/train.py --data data/clean.csv --register_as sales_nb
```

### Tests
```bash
pytest -q
```

### Docker
```bash
docker build -t sales-api .
docker run -p 8000:8000 -e MODEL_PATH=artifacts/nb_pipeline.joblib sales-api
```

### docker compose (API + MLflow)
```bash
docker compose up --build
```



