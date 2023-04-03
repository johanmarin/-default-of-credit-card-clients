
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier


def get_importances(X, y, classifier=RandomForestClassifier(), random_state=123) -> pd.DataFrame:
    """
    Calculate the feature importances of a classifier using the given X and y data.

    Args:
    - X (pd.DataFrame): The features data.
    - y (pd.Series): The target data.
    - classifier (object): The classifier object with a 'fit' method. Default is RandomForestClassifier.
    - random_state (int): The random state to be used for reproducibility. Default is 123.

    Returns:
    - pd.DataFrame: A dataframe with the feature importances for each variable.
    """
    clf_rf = RandomForestClassifier(random_state=random_state)      
    clf_rf.fit(X, y)
    classifier  = classifier.fit(X, y)
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]
    
    return pd.DataFrame.from_dict({"ind": indices, 
                                   "variable": X.columns, 
                                   "importance": importances, 
                                   "std": std}
                                  ).sort_values("ind").reset_index(drop=True).drop(columns="ind")
    

def get_feature_selection(X: pd.DataFrame, y: pd.Series, classifier=RandomForestClassifier(), scr: str='f1') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function performs feature selection on the given dataset using Recursive Feature Elimination with Cross-Validation (RFECV) and returns the selected features along with their importance ranking and the cross-validation results.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        classifier (estimator object, optional): The estimator object used to fit the model. Default is RandomForestClassifier().
        scr (str, optional): The scoring metric used for cross-validation. Default is 'f1'.

    Returns:
        tuple:
            - pd.DataFrame: The selected features along with their importance ranking.
            - pd.DataFrame: The cross-validation results.

    Example:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        selected_features, cv_results = get_feature_selection(X_train, y_train)
    """
    rfecv = RFECV(estimator=classifier, step=1, cv=5, scoring=scr)   #5-fold cross-validation
    rfecv = rfecv.fit(X, y)
    
    return pd.DataFrame.from_dict({"variable": X.columns, 
                                "include": rfecv.support_, 
                                "importance": rfecv.ranking_}
                            ).sort_values("importance"), pd.DataFrame.from_dict(rfecv.cv_results_)[["mean_test_score", "std_test_score"]]
    


def select_pca_n_components(X: pd.DataFrame, var_threshold: float=0.95) -> int:
    """
    Selects the minimum number of principal components to retain in order to explain at least var_threshold
    of the total variance.

    Args:
        X (pd.DataFrame): The feature matrix to apply PCA on.
        var_threshold (float): The proportion of total variance that should be explained by the selected components.
        Defaults to 0.95.

    Returns:
        int: The minimum number of principal components that should be retained to explain at least var_threshold
        of the total variance.
    """
    # Define PCA estimator and pipeline
    pca = PCA()
    pca_pipeline = Pipeline([('pca', pca)])
    
    # Fit PCA on feature matrix
    pca_pipeline.fit(X)
    
    # Get variance explained by each component
    var_exp = pca.explained_variance_ratio_
    
    # Calculate cumulative variance explained
    cum_var_exp = np.cumsum(var_exp)
    
    # Find number of components that explain at least var_threshold of the total variance
    n_components = np.argmax(cum_var_exp >= var_threshold) + 1

    return n_components   

def fit_pca(X: pd.DataFrame) -> PCA:
    """
    Fits a PCA model to a feature matrix X.

    Args:
        X (pd.DataFrame): A feature matrix of shape (n_samples, n_features)

    Returns:
        PCA: A fitted PCA object with n_components selected to explain at least 95% of the variance in X.
    """
    n_pca = select_pca_n_components(X)
    pca = PCA(n_components=n_pca)
    pca = pca.fit(X)
    return pca

    

def plot_importances(df_importances: pd.DataFrame):
    """
    A function to plot feature importances.

    Args:
        df_importances (pd.DataFrame): A DataFrame containing feature importances, sorted by importance.

    Returns:
        None, but generates a plot.
    """
    plt.style.use("ggplot") 
    vars = df_importances["variable"]
    importances = df_importances["importance"]
    std = df_importances["std"]
    indices = df_importances.index

    plt.figure(1, figsize=(14, 5))
    plt.title("Feature importances")
    plt.bar(range(len(vars)), importances[indices],
        color="g", yerr=std[indices], align="center")
    plt.xticks(range(len(vars)), vars[indices],rotation=90)
    plt.xlim([-1, len(vars)])
    plt.show()
    
def plot_number_features(feature_analisis: pd.DataFrame):
    """
    Plots the cross validation score of the number of selected features for a given feature selection analysis.

    Args:
    - feature_analisis (pd.DataFrame): A pandas DataFrame with the results of a feature selection analysis.
        It must contain the columns "mean_test_score" and "std_test_score" with the mean and standard deviation
        of the cross validation scores, respectively.

    Returns:
    - None.
    """
    plt.style.use("ggplot") 
    mean_scores = feature_analisis.mean_test_score
    std_scores = feature_analisis.std_test_score

    plt.figure(figsize=(14, 5))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, feature_analisis.shape[0]+1), mean_scores)

    # Fill area between upper and lower standard deviation bounds
    plt.fill_between(range(1, len(mean_scores)+1), 
                    mean_scores - std_scores, 
                    mean_scores + std_scores, 
                    alpha=0.2)
    plt.show()
    
def plot_pca_variance(X: pd.DataFrame):
    """
    Plots the explained variance ratio of Principal Component Analysis (PCA) for a given feature matrix X.

    Args:
        X: A pandas DataFrame containing the feature matrix.

    Returns:
        None.
    """
    pca = PCA()
    pca.fit(X)
    plt.style.use("ggplot") 
    plt.figure(1, figsize=(14, 5))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')
    
    