import pandas as pd
from src.split_data import split_dataset

def test_split_dataset_shapes():
    df = pd.DataFrame([
        {"a":1.0,"b":2.0,"c":"x","Sales_Classification":"yes"},
        {"a":2.0,"b":3.0,"c":"y","Sales_Classification":"no"},
        {"a":3.0,"b":4.0,"c":"x","Sales_Classification":"yes"},
        {"a":4.0,"b":5.0,"c":"y","Sales_Classification":"no"},
    ])
    xtr, xte, ytr, yte = split_dataset(df, target_col="Sales_Classification", test_size=0.25, random_state=1)
    assert len(xtr) + len(xte) == len(df)
    assert len(ytr) + len(yte) == len(df)
