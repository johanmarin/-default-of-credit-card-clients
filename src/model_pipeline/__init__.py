
from src.model_pipeline.feature_selection import get_importances, get_feature_selection, fit_pca
from src.model_pipeline.model_selection import evaluate_model
import src.utils as ut
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import klib
import os

import src.model_pipeline.time_ahor as vrt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


class DefaultModeler:
    """
    A class for building and evaluating machine learning models on a given dataset using default and feature-selected
    data, as well as PCA-transformed data.

    Args:
        file_path (str): The path to the dataset file.
        random_state (int): The random seed to use for reproducibility. Defaults to 123.

    Attributes:
        df (pandas.DataFrame): The preprocessed dataset as a pandas DataFrame.
        random_state (int): The random seed used for reproducibility.
        X (pandas.DataFrame): The input features of the dataset.
        y (pandas.Series): The target variable of the dataset.
        feature_filter (list): A list of selected features after feature selection.
        pca (PCA): A PCA transformer fitted on the input features.
        X_train (pandas.DataFrame): The training input features.
        X_test (pandas.DataFrame): The test input features.
        y_train (pandas.Series): The training target variable.
        y_test (pandas.Series): The test target variable.
        metadata_models (pandas.DataFrame): A DataFrame containing the evaluation metrics of different models.
        model (sklearn estimator): The best model based on the evaluation metrics.

    Methods:
        set_vars_training(self): Extracts the input features and target variable from the dataset for training.
        set_vars_use(self): Extracts the input features from the dataset for prediction, based on the required columns.
        fit_selectors(self): Applies feature selection and PCA transformation to the input features.
        use_selectors(self): Uses the previously fitted feature selectors to transform the input features.
        split_train_test(self): Splits the input features and target variable into training and test sets,
            and applies undersampling to the training set if needed.
        train_models(self): Trains different models on the training set with default and transformed input features,
            and selects the best model based on the evaluation metrics.

    Examples:
        # Instantiate the modeler and preprocess the dataset
        >>> modeler = DefaultModeler('path/to/dataset.csv')

        # Extract the input features and target variable for training
        >>> modeler.set_vars_training()

        # Apply feature selection and PCA transformation to the input features
        >>> modeler.fit_selectors()

        # Split the input features and target variable into training and test sets
        >>> modeler.split_train_test()

        # Train different models on the training set with default and transformed input features,
        # and select the best model based on the evaluation metrics
        >>> modeler.train_models()

        # Use the previously fitted feature selectors to transform the input features for prediction
        >>> modeler.use_selectors()

        # Extract the input features from the dataset for prediction, based on the required columns
        >>> modeler.set_vars_use()

        # Predict the target variable using the selected model
        >>> y_pred = modeler.model.predict(modeler.X)
    """


    
    def __init__(self, file_path: str, random_state = 123) -> None:  
        """
        Instantiate the DefaultModeler class
        
        Args:
        -----
        file_path: str
            Path to the csv file
        random_state: int
            Seed value for random initialization
        """
        self.df = klib.data_cleaning(pd.read_csv(file_path))        
        self.random_state= random_state
        
    def set_vars_training(self):
        """
        Set X, y variables for the training process
        """
        self.X = self.df.iloc[:,1:]
        self.y = self.df.iloc[:,0]
        
        ut.registry_object(list[self.X.columns], 'required_columns.joblib')
        
    def set_vars_use(self):
        """
        Set X variable for the model prediction process
        """
        columns = ut.get_registred_object('required_columns.joblib')
        self.X = self.df[columns]        
        
    def fit_selectors(self):        
        """
        Run feature selection and PCA algorithms and stores the results
        """
        self.df_importances = get_importances(self.X, self.y, random_state=self.random_state)
        
        self.selected_features, self.feature_analisis = vrt.ft ,vrt.an
        # self.selected_features, self.feature_analisis = get_feature_selection(self.X, self.y)
        self.pca= fit_pca(self.X)
        self.feature_filter = self.selected_features.variable[self.selected_features.include].to_list()
        
        selectors = {"filter": self.feature_filter,
                     "pca": self.pca}
        
        ut.registry_object(selectors, 'feature_selectors.joblib')
        
    def use_selectors(self):   
        """
        Retrieve the stored feature selection and PCA results for predicting using the model
        """             
        selectors = ut.get_registred_object('feature_selectors.joblib')
        self.feature_filter = selectors["filter"]
        self.pca = selectors["pca"]       
    
    def split_train_test(self):
        """Split the data into training and testing sets and performs random under-sampling.
        """        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=self.random_state)
        
        if (self.y.value_counts()/self.y.shape).equals(1/self.y.value_counts().shape[0]) == False:            
            rus = RandomUnderSampler(random_state=self.random_state)
            self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
            
    def train_models(self):        
        """Train models using default, filtered, and PCA transformed data.
        """
        models = [(RandomForestClassifier(random_state=self.random_state), 
            {'bootstrap': [True],
                'ccp_alpha': [0.0],
                'class_weight': [None],
                'criterion': ['gini'],
                'max_depth': [None],
                'max_features': ['sqrt'],
                'max_leaf_nodes': [None],
                'max_samples': [None],
                'min_impurity_decrease': [0.0],
                'min_samples_leaf': [1],
                'min_samples_split': [2],
                'min_weight_fraction_leaf': [0.0],
                'n_estimators': [20, 100],
                'n_jobs': [None],
                'oob_score': [False],
                'random_state': [None],
                'verbose': [0],
                'warm_start': [False]
                }),
            (GradientBoostingClassifier(random_state=self.random_state),
            {'ccp_alpha': [0.0],
                'criterion': ['friedman_mse'],
                'init': [None],
                'learning_rate': [0.1],
                'loss': ['log_loss'],
                'max_depth': [3],
                'max_features': [None],
                'max_leaf_nodes': [None],
                'min_impurity_decrease': [0.0],
                'min_samples_leaf': [1],
                'min_samples_split': [2],
                'min_weight_fraction_leaf': [0.0],
                'n_estimators': [100],
                'n_iter_no_change': [None],
                'random_state': [None],
                'subsample': [1.0],
                'tol': [0.0001],
                'validation_fraction': [0.1],
                'verbose': [0],
                'warm_start': [False]}),
            (KNeighborsClassifier(),
            {'algorithm': ['auto'],
                'leaf_size': [30],
                'metric': ['minkowski'],
                'metric_params': [None],
                'n_jobs': [None],
                'n_neighbors': [5],
                'p': [2],
                'weights': ['uniform']})  
            ]
        
        self.metadata_models = pd.DataFrame()
        
        for data in ["default", "filter", "pca"]:
            if data == "default":
                X_train = self.X_train
                X_test = self.X_test
            elif data == "filter":
                X_train = self.X_train[self.feature_filter]
                X_test = self.X_test[self.feature_filter]
            elif data == "pca":
                X_train = self.pca.transform(self.X_train)
                X_test = self.pca.transform(self.X_test)
            
            self.metadata_models = pd.concat([self.metadata_models,
                    pd.DataFrame([evaluate_model(est, params, data, X_train, self.y_train, X_test, self.y_test, self.random_state) for est, params in models])])
            
        self.model = self.metadata_models[self.metadata_models.f1_test == self.metadata_models.f1_test.max()].T[0]
            
        ut.registry_object(self.metadata_models, 'models_log.joblib')
        ut.registry_object(self.model, 'model.joblib')        
        
    def use_model(self):   
        """Uses the trained model to predict on new data.
        """     
        self.model = ut.get_registred_object('model.joblib')
        
        if self.model.data == "default":
            X = self.X
        elif self.model.data == "filter":
            X = self.X[self.feature_filter]
        elif self.model.data == "pca":
            X = self.pca.transform(self.X)
            
        self.predictions = self.model.model.predict(X)
        
    def get_log_training(self) -> pd.DataFrame:
        """Retrieve the log of model training.

        Args:
            None

        Returns:
            metadata_models (pd.DataFrame): DataFrame containing the log of model training.
        """
        self.metadata_models = ut.get_registred_object('models_log.joblib')
        return self.metadata_models
    
    def get_info_model(self) -> dict:
        """Retrieve the information of the trained model.

        Args:
            None

        Returns:
            model_dict (dict): Dictionary containing the information of the trained model.
        """
        self.model = ut.get_registred_object('model.joblib')
        return self.model.to_dict()