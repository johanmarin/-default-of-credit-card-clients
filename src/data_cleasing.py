
import pandas as pd

def remove_weird_values(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    return df[df[metadata.keys()].apply(lambda col: col.isin(metadata[col.name].keys()), axis = 0).sum(axis=1)==3] 