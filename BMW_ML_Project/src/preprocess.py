from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def _onehot_encoder():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

def build_preprocessor(
    scaled_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('minmax', MinMaxScaler(), scaled_cols),
            ('onehot', _onehot_encoder(), categorical_cols),
        ],
        remainder='drop'
    )
