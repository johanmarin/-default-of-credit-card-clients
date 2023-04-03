from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import multiprocessing
import pandas as pd

def test_hiperparametros(estimator, param_grid, X_train, y_train, data: str="default", random_state: int=123):
    """
    Performs hyperparameter tuning using GridSearchCV and returns the best model and its evaluation metrics.

    Args:
        estimator (object): A machine learning estimator object.
        param_grid (dict): A dictionary containing the hyperparameters and their potential values to be tuned.
        X_train (array-like): The training data samples.
        y_train (array-like): The target values.
        data (str): A string that describes the data being used for evaluation. Default is "default".
        random_state (int): An integer to set the random seed for reproducibility. Default is 123.

    Returns:
        pandas.Series: A pandas Series object containing the following evaluation metrics of the best model:
            - "data" (str): The description of the data used for evaluation.
            - "estimator" (object): The estimator used for the evaluation.
            - "model" (object): The best model selected by GridSearchCV.
            - "metric" (str): The evaluation metric used to select the best model. In this case, it is "f1".
            - "score" (float): The score of the best model on the evaluation metric.
            - "params" (dict): The hyperparameters of the best model.
    """
    grid = GridSearchCV(
            estimator  = estimator,
            param_grid = param_grid,
            scoring    = 'f1',
            n_jobs     = multiprocessing.cpu_count() - 1,
            cv         = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state), 
            refit      = True,
            verbose    = 0,
            return_train_score = True
            )
    grid.fit(X = X_train, y = y_train) 
    return pd.Series([data, estimator, grid.best_estimator_, grid.scoring, grid.best_score_, grid.best_params_],
        index = ["data", "estimator", "model", "metric", "score", "params"])


def evaluate_model(estimator, param_grid, data, X_train, y_train, X_test, y_test, random_state: int=123):
    """
    Evaluates a model using hyperparameters tuning and calculates the performance metrics including precision, recall and F1-score for both training and test sets.

    Args:
        estimator (sklearn estimator object): The estimator object to use for hyperparameters tuning.
        param_grid (dict): The parameter grid to search for the best hyperparameters using GridSearchCV.
        data (str): The name of the dataset being used for evaluation.
        X_train (pandas.DataFrame or numpy.ndarray): The feature matrix of the training set.
        y_train (pandas.Series or numpy.ndarray): The target vector of the training set.
        X_test (pandas.DataFrame or numpy.ndarray): The feature matrix of the test set.
        y_test (pandas.Series or numpy.ndarray): The target vector of the test set.
        random_state (int): Random seed to use for reproducibility.

    Returns:
        pandas.Series: A pandas series object containing the evaluation results including best model, best parameters, best score precision, recall and F1-score for both training and test sets.
    """  
    d = test_hiperparametros(estimator, param_grid, X_train, y_train, data, random_state)
    
    d.model = d.model.fit(X_train, y_train)    
    d["precision_train"], d["recall_train"], d["f1_train"], _ = precision_recall_fscore_support(y_train, d.model.predict(X_train), average='binary')
    d["precision_test"], d["recall_test"], d["f1_test"], _ = precision_recall_fscore_support(y_test, d.model.predict(X_test), average='binary')
    return d