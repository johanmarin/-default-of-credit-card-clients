# Model for default prediction

## Process
This repository implements a model for default prediction. Taking into account multiple transactional characteristics of each customer, it seeks to determine whether they will pay the next month or not.

The framework is detailed below:

1. Understanding the problem
2. Data quality analysis
3. Data quality correction
4. Exploratory analysis
5. Feature engineering
6. Variable selection
7. Experiments for selecting the best model (metric selection, hyperparameter tuning)
8. Model selection
9. Code refactoring (code modularization, class construction, docstring and annotations)
10. Building data pipelines and model training
11. Model registry.

## Notes
Custom libraries are built that allow for rapid retraining and deployment of the best model. The code implementation was made to be independent of the data set and variable names. The pipeline receives the required data for its configuration from a YAML file. Therefore, with small changes in the configuration, it can be used to build models aimed at different objectives.
  

## Development

```bash
.
├── registry                           <- Folder with model binaries, scalers, PCA, etc.
│   ├── data_pipeline                  <- Data pipeline binaries.
│   └── model_pipeline                 <- Model pipeline binary.
│
├── src                                <- Modules for data cleaning, data exploration, and model fitting.
│   ├── data_pipeline                  <- Data pipeline modules.
│   │   ├── __init__.py                <- Contains the main Class of the data pipeline that allows it to be executed.
│   │   ├── data_cleansing.py          <- Functions used in data cleaning.
│   │   └── feature_engineering.py     <- Functions used in feature engineering.
│   ├── model_pipeline                 <- Model pipeline modules.
│   │   ├── __init__.py                <- Contains the main Class of the model pipeline that allows it to be executed.
│   │   ├── feature_selection.py       <- Functions used in feature selection.
│   │   └── model_selection.py         <- Functions for training and evaluating models.
│   ├── exploratory.py                 <- Functions for data visualization.
│   ├── utils                          <- Functions for saving and reading variables in the registry.
│   └── __init__.py                    <- File that is read first when importing the datasets folder.
│
└── requirements.txt                   <- File with the versions of the necessary packages.
                           
```   

## Configuration file

The configuration file should have information about the dataset and some variables on how it will be processed.

```yaml
path: https://url/to data # This is the URL path to the dataset
categories: # Categorical variables with non-interpretable values are used for graphics
  name_var1:
    'value1': Interpretation of value1  
    'value2': Interpretation of value2
  name_var2:
    'value1': Interpretation of value1
    'value2': Interpretation of value2
    'value3': Interpretation of value3
dtypes:  # Dtypes for all values are used to guarantee the correct processing of variables
  name_var1: category
  name_var2: category
  name_var3: int64
  name_var4: int64
  name_var5: int64
  name_target: bool

target: name_target  # Name for the target variable

rename:   # If required, rename a variable
  name_var: new_name_var 

regex:
  slope: r"pay_[2-6]+"  # Use this regex to select variable names if I need to calculate slope
  relatives: r".*amt"  # Use this regex to select variable names if any variable is better in relative format
  total: LIMIT_BAL  # Variable that divides variables and transforms them into relatives
    
```

## Analysis
To test the code and perform the analysis, the default of credit card clients Data Set is used, stored in this [repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). The analysis is carried out using the framework detailed above in the following files.

```bash
.
├── analysis.ipynb     <- Consolidates the entire framework detailed above. It uses the modules and functions described 
│                         below. Additionally, it contains comments and details the process.
└── metadata.yaml      <- Configuration values of the database used for modeling.

``` 