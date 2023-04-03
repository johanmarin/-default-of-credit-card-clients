from src.data_pipeline.feature_enginering import calculate_slope_df, fit_scaler, scale_data
from src.data_pipeline.data_cleasing import remove_weird_values
import src.utils as ut
import pandas as pd
import yaml
import klib
import re
import os

class DefaulerData:
    """
    Class for preparing and cleaning data for the credit default model.

    Args:
        str_url (str): URL for the excel file containing the data
        str_metadata_path (str): Path for the metadata file containing data types
        int_sample (int, optional): Number of samples to include. Defaults to None.
        str_type (str, optional): Type of data ('train' or 'test'). Defaults to 'train'.

    Attributes:
        type (str): Type of data ('train' or 'test')
        metadata (dict): Dictionary containing metadata for the dataset
        df (pd.DataFrame): Cleaned and preprocessed dataframe

    Methods:
        cleasing_data(): Cleans data by removing values that do not align with metadata
        build_features(): Adds engineered features to the dataframe
        train_scaler(): Scales numerical columns and saves the scaler objects
        use_scaler(): Uses the scaler objects to transform numerical columns
        encoding_catagories(): Encodes categorical columns as one-hot encoded columns
        save_columns(): Saves column names to be used later
        get_columns(): Retrieves column names
        save_data(): Saves the cleaned dataframe as a csv file

    """
    def __init__(self, str_url: str, str_metadata_path: str, int_sample: int=None, str_type: str="train") -> None:
        """
        Initializes the DefaulerData object and loads the data and metadata.

        Args:
            str_url (str): URL for the excel file containing the data
            str_metadata_path (str): Path for the metadata file containing data types
            int_sample (int, optional): Number of samples to include. Defaults to None.
            str_type (str, optional): Type of data ('train' or 'test'). Defaults to 'train'.

        Raises:
            ValueError: Raised when input data is not provided
        """
        self.type = str_type
        # load metadata
        file = open(str_metadata_path, "r")
        metadata = yaml.safe_load(file)
        file.close()
        
        # load data
        df = pd.read_excel(str_url, header=1, index_col="ID", dtype=metadata["dtypes"]).rename(columns={'PAY_0': 'PAY_1'})
        if int_sample:
            df = df.sample(n=int_sample, random_state=1)
        
        cols = list(df.columns)
        klib.clean_column_names(df)
        str_meta = str(metadata)
        for i in range(len(cols)):
            str_meta = str_meta.replace(cols[i], df.columns[i])
        self.metadata = eval(str_meta)       
        
        df = df.apply(lambda col: col.astype(str).astype("category") if str(col.dtype) == "category" else col, axis=0)
        self.df = klib.data_cleaning(df)
        
    def cleasing_data(self):
        """
        Cleans data by removing values that do not align with metadata
        """
        self.df = remove_weird_values(self.df, self.metadata["categories"])
        
    def build_features(self):
        """
        Adds engineered features to the dataframe
        """
        self.df["slope"] = calculate_slope_df(self.df, [col for col in self.df.columns if re.search(r"pay_[2-6]+", col)])
        
        for col in [col for col in self.df.columns if re.search(r".*amt", col)]:
            self.df[f"percent_{col}"] = self.df[col]/self.df.limit_bal
    
    def train_scaler(self):   
        """
        Scales numerical columns and saves the scaler objects
        """
        scalers = {}
        for col_name in self.df.select_dtypes(include=['integer', 'floating']).columns.tolist():
            scalers[col_name], self.df[f"{col_name}_scaled"] = fit_scaler(data=self.df[col_name])
            self.df.drop(columns=col_name, inplace=True)
        
        ut.registry_object(scalers, 'scalers.joblib')
        self.df = klib.data_cleaning(self.df)
            
    def use_scaler(self):
        """
        Uses the scaler objects to transform numerical columns
        """
        
        if os.path.exists('scalers.joblib'):
            scalers = ut.get_registred_object('scalers.joblib')
        else:
            raise ValueError(" file scalers.joblib not exists, you need train the model before using this")
        
        for col_name in self.df.select_dtypes(include=['integer', 'floating']).columns.tolist():
            self.df[f"{col_name}_scaled"] = scale_data(self.df[col_name], scalers[col_name])
            self.df.drop(columns=col_name, inplace=True)


    def encoding_catagories(self):
        """
        Encodes categorical columns as one-hot encoded columns
        """
        if os.path.exists('columns_data.joblib'):            
            self.df = pd.get_dummies(self.df)
        else:
            self.df = pd.get_dummies(self.df, drop_first = True)
        self.df = klib.data_cleaning(self.df)
        
        
    def save_columns(self):
        """
        Saves column names to be used later
        """
        self.df.columns = [self.metadata["target"]] + [col for col in self.df.columns if col != self.metadata["target"]]
        ut.registry_object(self.df.columns, "columns_data.joblib")
        
    def get_columns(self):
        """
        Retrieves column names
        """
        if os.path.exists('columns_data.joblib'):
            self.df = self.df[ut.get_registred_object("columns_data.joblib")]
        else:
            raise ValueError(" file columns_data.joblib not exists, you need train the model before using this")
        
    def save_data(self):
        """
        Saves the cleaned dataframe as a csv file
        """
        self.df.to_csv("data_model.csv", index=False)