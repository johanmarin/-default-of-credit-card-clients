import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calculate_slope(values: np.array) -> float:
    """
    Calculates the slope of the linear regression line for a given set of values.

    Args:
        values (np.array): A 1-D numpy array of numerical values.

    Returns:
        float: The slope of the linear regression line.

    Raises:
        ValueError: If values is not a 1-D numpy array.

    Examples:
        >>> values = np.array([1, 2, 3, 4, 5])
        >>> calculate_slope(values)
        1.0
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1-D numpy array")

    indices = np.arange(len(values))
    slope, intercept = np.polyfit(indices, values, 1)

    return slope


import pandas as pd

def calculate_slope_df(df: pd.DataFrame, name_columns: list) -> pd.Series:
    """
    Calculates the slope of the linear regression line for each row in a subset of columns in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be used for the calculation.
        name_columns (list): A list of column names to be used for the calculation.

    Returns:
        pd.Series: A pandas Series containing the slope of the linear regression line for each row.

    Raises:
        ValueError: If name_columns is not a list or if any column in name_columns is not present in df.

    Examples:
        >>> df = pd.DataFrame({
        ...     'column1': [1, 2, 3],
        ...     'column2': [4, 5, 6],
        ...     'column3': [7, 8, 9]
        ... })
        >>> name_columns = ['column1', 'column2']
        >>> calculate_slope_df(df, name_columns)
        0    1.0
        1    1.0
        2    1.0
        dtype: float64
    """
    if not isinstance(name_columns, list):
        raise ValueError("name_columns must be a list")
    if not all(col in df.columns for col in name_columns):
        raise ValueError("all columns in name_columns must be present in df")

    return df[name_columns].apply(lambda row: calculate_slope(row.values), axis=1)


def fit_scaler(data: pd.Series = None, scaler_type: str = "standard") -> tuple[object, np.array]:
    """
    Fits and returns a scaler object to transform input data based on the specified scaler type.

    Args:
        data (pd.Series): A pandas Series containing the input data.
        scaler_type (str): The type of scaler to be used. Choose from 'standard' or 'min_max'.
            Default is 'standard'.

    Returns:
        tuple[object, np.array]: A tuple containing the scaler object and the transformed data.

    Raises:
        ValueError: If data is None or if scaler_type is not 'standard' or 'min_max'.

    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> scaler_type = 'standard'
        >>> scaler, scaled_data = fit_scaler(data, scaler_type)
        >>> scaled_data
        array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
    """
    if data is None:
        raise ValueError("Input data is required")

    SCALER_MAP = {
        'standard': StandardScaler,
        'min_max': MinMaxScaler,
    }

    scaler_class = SCALER_MAP.get(scaler_type)
    if scaler_class is None:
        raise ValueError("Invalid scaler type. Choose 'standard' or 'min_max'.")

    scaler = scaler_class()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaler, scaled_data.flatten()


def scale_data(series_data: pd.Series, scaler) -> pd.DataFrame:
    """
    Scales input data using a scaler object and returns a flattened numpy array.

    Args:
        series_data (pd.Series): A pandas Series containing the input data to be scaled.
        scaler (object): A scaler object to be used for scaling the input data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the scaled and flattened input data.

    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> scaler = StandardScaler()
        >>> scaled_data = scale_data(data, scaler)
        >>> scaled_data
            0   -1.414214
            1   -0.707107
            2    0.000000
            3    0.707107
            4    1.414214
            dtype: float64
    """
    # Use the scaler object to transform the input data and flatten the resulting numpy array
    scaled_data = scaler.fit_transform(series_data.values.reshape(-1, 1)).flatten()

    # Create a pandas DataFrame containing the scaled data and return it
    return pd.DataFrame(scaled_data) 

