import pandas as pd
from src.preprocess import build_preprocessor

def test_preprocessor_output_shape():
    scaled = ['Mileage_KM','Price_USD','Sales_Volume']
    cats = ['Fuel_Type','Color','Region','Model','Transmission']
    pre = build_preprocessor(scaled, cats)
    df = pd.DataFrame([{
        "Mileage_KM": 1.0, "Price_USD": 2.0, "Sales_Volume": 3.0,
        "Fuel_Type": "Petrol", "Color": "White", "Region": "MENA", "Model": "Sedan", "Transmission": "Manual"
    }])
    X = pre.fit_transform(df)
    assert X.shape[0] == 1
    assert X.shape[1] >= len(scaled)
