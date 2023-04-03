
import pandas as pd

def remove_weird_values(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Removes rows from a pandas DataFrame that contain values not specified in metadata.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.
        metadata (dict): A dictionary containing metadata for each column in df.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows with valid values.

    Raises:
        ValueError: If metadata does not contain keys for all columns in df.

    Examples:
        >>> metadata = {
        ...     'column1': {1: 'valid', 2: 'valid', 3: 'invalid'},
        ...     'column2': {'a': 'valid', 'b': 'invalid', 'c': 'valid'},
        ...     'column3': {'x': 'valid', 'y': 'invalid', 'z': 'valid'}
        ... }
        >>> df = pd.DataFrame({
        ...     'column1': [1, 2, 3],
        ...     'column2': ['a', 'b', 'c'],
        ...     'column3': ['x', 'y', 'z']
        ... })
        >>> remove_weird_values(df, metadata)
           column1 column2 column3
        0        1       a       x
        2        3       c       z
    """
    return df[df[metadata.keys()].apply(lambda col: col.isin(metadata[col.name].keys()), axis=0).sum(axis=1) == len(metadata)]
