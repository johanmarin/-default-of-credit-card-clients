import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

def dataframe_unanonimized(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Replaces the anonymous values in a dataframe with their original values based on metadata.

    Args:
        df: A pandas dataframe with anonymous values.
        metadata: A dictionary containing information about the anonymous values.
            The dictionary must have a "categories" key that maps column names to a dictionary of 
            anonymous values and their original values.

    Returns:
        A pandas dataframe with anonymous values replaced with their original values.
    """
    return df.apply(lambda col: col.replace(metadata["categories"][col.name]) 
                    if str(col.name) in metadata["categories"] 
                    else col, axis=0)

def generate_profiling_report(df: pd.DataFrame, metadata: dict, file_name: str) -> pandas_profiling.pandas_profiling.ProfileReport:
    """
    Generate a profiling report for the input pandas DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame to generate a profiling report for.
    metadata (dict): A dictionary containing metadata for the DataFrame.
    file_name (str): The name of the output file for the generated report.

    Returns:
    pandas_profiling.pandas_profiling.ProfileReport: A profile report object for the input DataFrame.

    Config:
    config (dict): A dictionary containing configuration options for the pandas_profiling.ProfileReport object.
    """
    config = {
        "minimal": True,
        "title": "Default credit card clients", 
        "dataset": {"description": "", 
                    "creator": "", 
                    "author": "", 
                    "copyright_holder": "", 
                    "copyright_year": "", 
                    "url": ""}, 
        "variables": {"descriptions": {}}, 
        "infer_dtypes": True, 
        "show_variable_description": False, 
        "pool_size": 0, 
        "progress_bar": True, 
        "vars": {"num": {"quantiles": [0.05, 0.25, 0.5, 0.75, 0.95], 
                        "skewness_threshold": 20, 
                        "low_categorical_threshold": 5, 
                        "chi_squared_threshold": 0.999,
                        "common_values": False}, 
                "cat": {"length": True, 
                        "characters": True, 
                        "words": True, 
                        "cardinality_threshold": 50, 
                        "imbalance_threshold": 0.5, 
                        "n_obs": 5, 
                        "chi_squared_threshold": 0.999, 
                        "coerce_str_to_date": False, 
                        "redact": False, 
                        "histogram_largest": 50, 
                        "stop_words": []}, 
                "image": {"active": False, 
                        "exif": True, 
                        "hash": True}, 
                "bool": {"n_obs": 3, 
                        "imbalance_threshold": 0.5, 
                        "mappings": {"t": True, "f": False, "yes": True, "no": False, "y": True, "n": False, "True": True, "False": False}}, "path": {"active": False}, 
                "file": {"active": False}, 
                "url": {"active": False}, 
                "timeseries": {"active": False, 
                                "sortby": None, 
                                "autocorrelation": 0.7, 
                                "lags": [1, 7, 12, 24, 30], 
                                "significance": 0.05, "pacf_acf_lag": 100}}, 
        "sort": None, 
        "missing_diagrams": {"bar": df.isnull().sum().sum() > 0, "matrix": df.isnull().sum().sum() > 0, "heatmap": df.isnull().sum().sum() > 0}, 
        "correlations": {"auto": 
            {"key": "auto", 
            "calculate": True, 
            "warn_high_correlations": 10, 
            "threshold": 0.5, 
            "n_bins": 10}}, 
        "correlation_table": True, 
        "interactions": {"continuous": False, 
                        "targets": [col for col, dtype in metadata["dtypes"].items() if not dtype in ["bool", "category"]][:1]}, 
        "categorical_maximum_correlation_distinct": df.shape[0]/10, 
        "memory_deep": False, 
        "plot": {"missing": {"force_labels": True, 
                            "cmap": "RdBu"}, 
                "image_format": "svg", 
                "correlation": {"cmap": "RdBu", "bad": "#000000"}, 
                "dpi": 800, "histogram": {"bins": 50, "max_bins": 250, "x_axis_labels": True}, 
                "scatter_threshold": 1000, 
                "cat_freq": {"show": True, "type": "bar", "max_unique": 10, "colors": None}}, 
        "duplicates": {"head": 10, "key": "# duplicates"}, 
        "samples": {"head": 10, "tail": 0, "random": 0}, 
        "reject_variables": True, 
        "n_obs_unique": 10, 
        "n_freq_table_max": 10, 
        "n_extreme_obs": 10, 
        "report": {"precision": 8}, 
        "html": {"style": {"primary_colors": ["#377eb8", "#e41a1c", "#4daf4a"], 
                        "logo": "", 
                        "theme": None}, 
                "navbar_show": True, 
                "minify_html": True, 
                "use_local_assets": True, 
                "inline": True, 
                "assets_prefix": None, 
                "assets_path": None, 
                "full_width": False}, 
        "notebook": {"iframe": {"height": "800px", "width": "100%", "attribute": "srcdoc"}}}

    report = df.profile_report(**config)
    report.to_file(file_name)

def visualize_boxplots(df: pd.DataFrame, target: str):
    """
    Create boxplots for all numeric features in the dataframe against the target variable.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        target (str): Target variable to compare against.

    Returns:
        None
    """
    plt.style.use("ggplot") 
    # Set color palette
    colors = ["red", "pink"]
    sns.set_palette(sns.color_palette(colors))
    
    # Create a list of all numeric columns in the dataframe except the target variable
    features = df.select_dtypes(include=['integer', 'floating']).columns.tolist()
    if target in features:
        features.remove(target)
        
    num_features = len(features)
    num_cols = min(num_features, 4)
    num_rows = math.ceil(num_features / num_cols)

    # Create a grid of subplots based on the number of variables in the dataframe
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    # Loop through each feature and create a boxplot against the target variable
    for i, feature in enumerate(features):
        row = i // num_cols
        col = i % num_cols
        sns.boxplot(x=target, y=feature, data=df, ax=axs[row, col])
        if num_rows > 1 and num_cols > 1:
            axs[row, col].set_title(f"{feature} vs {target}")
        else:
            axs[i].set_title(f"{feature} vs {target}")

    plt.tight_layout()
    plt.show()

    
def visualize_stacked_barplots(df: pd.DataFrame, target: pd.DataFrame):
    """
    Visualizes stacked barplots of categorical variables against a target variable.

    Args:
    df (pd.DataFrame): The dataframe to visualize.
    target (pd.DataFrame): The target variable in the dataframe.

    Returns:
    None

    Raises:
    None
    """
    # Create a list of all categorical columns in the dataframe except the target variable
    plt.style.use("ggplot") 
    cat_cols = df.select_dtypes(include=['category', 'bool', 'object']).columns.tolist()
    if target in cat_cols:
        cat_cols.remove(target)
        
    num_cols = min(len(cat_cols), 4)
    num_rows = math.ceil(len(cat_cols) / num_cols)

    # Create a grid of subplots based on the number of categorical variables in the dataframe
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    # Loop through each categorical variable and create a stacked bar plot against the target variable
    for i, cat_col in enumerate(cat_cols):
        row = i // num_cols
        col = i % num_cols        
        # Calculate the group counts for the categorical variable and target variable
        group_counts = pd.crosstab(df[cat_col], df[target], normalize='index')
        # Add Title and Labels
        if num_rows > 1 and num_cols > 1:
            group_counts.plot(kind='bar', stacked=True, color=['red', 'pink'], ax=axs[row, col])
            axs[row, col].set_title(f"{cat_col} vs {target}")
        else:
            group_counts.plot(kind='bar', stacked=True, color=['red', 'pink'], ax=axs[i])
            axs[i].set_title(f"{cat_col} vs {target}")
    plt.tight_layout()
    plt.show()


def boxplot_data_with_outliers(data: pd.Series, threshold: int=3):
    """
    Visualize a boxplot of the data with and without outliers.

    Args:
        data (pd.Series): A pandas series of numerical data.
        threshold (int, optional): The number of standard deviations from the mean to consider a point as an outlier. Defaults to 3.

    Returns:
        None.
    """
    # calculate the mean and standard deviation of the data
    plt.style.use("ggplot") 
    mean = np.mean(data)
    std = np.std(data)
    
    # calculate the z-score for each data point
    z_scores = [(x - mean) / std for x in data]
    
    # identify the outliers based on the threshold value
    outliers = [x for i, x in enumerate(data) if abs(z_scores[i]) > threshold]
    
    # create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # create a boxplot of the data with outliers in the first subplot
    ax1.boxplot(data)
    ax1.set_title('Data with Outliers')
    
    # create a boxplot of the data without outliers in the second subplot
    data_without_outliers = [x for x in data if x not in outliers]
    ax2.boxplot(data_without_outliers)
    ax2.set_title('Data without Outliers')
    
    # display the plot
    plt.show()
    
def calculate_outlier_percentage_zscore(data: pd.Series, threshold: int=3) -> float:
    """
    Calculates the percentage of outliers in the given data using the z-score method.

    Args:
        data: A pandas Series of numerical data.
        threshold: The z-score threshold for identifying outliers. Defaults to 3.

    Returns:
        The percentage of outliers in the data as a float.

    Raises:
        None.
    """
    # calculate the mean and standard deviation of the data
    mean = np.mean(data)
    std = np.std(data)
    
    # calculate the z-score for each data point
    z_scores = [(x - mean) / std for x in data]
    
    # identify the outliers based on the threshold value
    outliers = [x for i, x in enumerate(data) if abs(z_scores[i]) > threshold]
    
    # calculate the percentage of outliers in the data
    outlier_percentage = (len(outliers) / len(data)) * 100
    
    # return the percentage of outliers
    return outlier_percentage

def analize_outliers_zscore(df: pd.DataFrame):
    """
    Analyzes outliers in a given pandas DataFrame using the z-score method and plots a boxplot with outliers.

    Args:
        df (pd.DataFrame): Pandas DataFrame to analyze outliers.

    Returns:
        None
    """
    for col in df.select_dtypes(include=['integer', 'floating']).columns.tolist():
        print(f"{col} have a {calculate_outlier_percentage_zscore(df[col])} % outliers")
        boxplot_data_with_outliers(df[col])