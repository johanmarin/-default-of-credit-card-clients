import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calculate_slope(values: np.array) -> float:    
    # create a numpy array of the Series indices
    indices = np.arange(len(values))
    
    # calculate the slope of the linear regression line
    slope, intercept = np.polyfit(indices, values, 1)
    
    return slope


def calculate_slope_df(df: pd.DataFrame, name_columns: list) -> pd.Series:
    return df[name_columns].apply(lambda row: calculate_slope(row.values), axis=1)

def fit_scaler(data: pd.Series = None, scaler_type: str = "standard") -> tuple[object, np.array]:
    if data is None:
        raise ValueError("Input data is required")
        
    SCALER_MAP = {
        'standard': StandardScaler,
        'min_max': MinMaxScaler,
    }

    # get the scaler class based on the specified scaler type
    scaler_class = SCALER_MAP.get(scaler_type)
    if scaler_class is None:
        raise ValueError("Invalid scaler type. Choose 'standard' or 'min_max'.")

    scaler = scaler_class()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaler, scaled_data.flatten()


def scale_data(series_data: pd.Series, scaler) -> pd.DataFrame:   
    # get the series data as a numpy array
    return scaler.fit_transform(series_data.values.reshape(-1, 1)).flatten()
