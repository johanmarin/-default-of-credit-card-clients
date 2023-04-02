from src.feature_enginering import calculate_slope_df, fit_scaler, scale_data
import src.utils as ut
from src.data_cleasing import remove_weird_values
import pandas as pd
import yaml
import klib
import re
import os

class DefaulerData:
    """
    A class for loading and processing default payment data.

    Args:
        str_url (str): The URL or path to the Excel file containing the data.
        str_metadata_path (str): The path to the YAML file containing the metadata.

    Attributes:
        metadata (dict): A dictionary containing the metadata for the data.
        df (pandas.DataFrame): A DataFrame containing the loaded data.
        int_sample(int): The number of rows that you need if is a sample

    Raises:
        FileNotFoundError: If the metadata file could not be found.

    """

    def __init__(self, str_url: str, str_metadata_path: str, int_sample: int=None, str_type: str="train") -> None:
        """
        Initializes a DefaulerData object.

        Args:
            str_url (str): The URL or path to the Excel file containing the data.
            str_metadata_path (str): The path to the YAML file containing the metadata.

        Raises:
            FileNotFoundError: If the metadata file could not be found.

        """
        self.type = str_type
        # load metadata
        file = open(str_metadata_path, "r")
        self.metadata = yaml.safe_load(file)
        file.close()
        
        # load data
        df = pd.read_excel(str_url, header=1, index_col="ID", dtype=self.metadata["dtypes"]).rename(columns={'PAY_0': 'PAY_1'})
        if int_sample:
            df = df.sample(n=int_sample, random_state=1)
        df = df.apply(lambda col: col.astype(str).astype("category") if str(col.dtype) == "category" else col, axis=0)
        # set string type to categorical variables
        self.df = klib.data_cleaning(df)
        self.df.columns = df.columns
        del(df)
        
    def cleasing_data(self):
        self.df = remove_weird_values(self.df, self.metadata["categories"])
        
    def build_features(self):
        self.df["slope"] = calculate_slope_df(self.df, [col for col in self.df.columns if re.search(r"PAY_[2-6]+", col)])
        
        for col in [col for col in self.df.columns if re.search(r".*AMT", col)]:
            self.df[f"percent_{col}"] = self.df[col]/self.df.LIMIT_BAL
    
    def train_scaler(self):   
        scalers = {}
        for col_name in self.df.select_dtypes(include=['integer', 'floating']).columns.tolist():
            scalers[col_name], self.df[f"{col_name}_scaled"] = fit_scaler(data=self.df[col_name])
            self.df.drop(columns=col_name, inplace=True)
        
        ut.registry_object(scalers, 'scalers.joblib')
        
        self.df = klib.data_cleaning(self.df)
        cols = self.df.columns
        self.df.columns = cols
        del(cols)
            
    def use_scaler(self):
        
        if os.path.exists('scalers.joblib'):
            scalers = ut.get_registred_object('scalers.joblib')
        else:
            raise ValueError(" file scalers.joblib not exists, you need train the model before using this")
        
        for col_name in self.df.select_dtypes(include=['integer', 'floating']).columns.tolist():
            self.df[f"{col_name}_scaled"] = scale_data(self.df[col_name], scalers[col_name])
            self.df.drop(columns=col_name, inplace=True)
        
        self.df = klib.data_cleaning(self.df)
        cols = self.df.columns
        self.df.columns = cols
        del(cols)

    def encoding_catagories(self):
        self.df = pd.get_dummies(self.df)
        self.df = klib.data_cleaning(self.df)
        
        cols = self.df.columns
        self.df.columns = cols
        del(cols)
        
    def save_columns(self):
        self.df.columns = [self.metadata["target"]] + [col for col in self.df.columns if col != self.metadata["target"]]
        ut.registry_object(self.df.columns, "columns_data.joblib")
        
    def get_columns(self):
        self.df = self.df[ut.get_registred_object("columns_data.joblib")]
        
    def save_data(self):
        self.df.to_csv("data_model.csv", index=False)