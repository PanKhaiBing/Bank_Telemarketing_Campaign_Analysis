# Data manipulation
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Functions and metrics
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score,f1_score, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor 

# Constants
TITLE_FONT = 16
LABEL_FONT = 12

## Step 1 to 5 representing the process within the main script

## Step 1: Load the dataset from a specified file path
def load_data(file_path):
    """
    Arguments:
        file_path: The location or the file path of the dataset.

    Purpose:
    Load the dataset from a specified file path.   
    """
    return pd.read_csv(file_path)

## Step 2: Inspect the data
# Check the class balance within the target variable
def check_class_proportion(data, target_col):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.
        target_col:          Target column to inspect.

    Purpose:
    Inspect the proportion of class within target column.
    """
    print(f'Proportion of the class in "{target_col}":')
    print(data[target_col].value_counts(normalize=True))
    print()

## Step 3: Data Cleaning
# A function to drop duplicated rows
def drop_duplicates(data):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.

    Purpose:
    Drop duplicated rows within DataFrame. 
    """
    # Drop duplicated rows
    data = data.drop_duplicates()
    # Reset the index of the resulting DataFrame
    return data.reset_index(drop=True)

# Rename columns by replacing dot with underscore 
def rename_column(data):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.

    Purpose:
    Renaming columns by replacing dot with underscore to improve readability. 
    """
    data.columns = data.columns.str.replace('.', '_', regex=False)
    return data

# A function to check the 'unknown' proportion and remove 'unknown' value by rows
def check_and_remove_unknown(data):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.

    Purpose:
    1. Check the proportion of 'unknown' values within the 'object' columns.
    2. Remove rows containing 'unknown' values within these columns.

    Returns:
    DataFrame with rows containing 'unknown' values removed.
    """
    print('Proportion of "unknown" within "object" columns:')
        
    # Track the initial row count
    total_rows = len(data)
        
    # Iterate over object columns
    for col in data.select_dtypes(include=['object']).columns:
        # Calculate proportion of 'unknown'
        unknown_count = (data[col] == 'unknown').mean()
        # Print proportion if greater than 0
        if unknown_count > 0:
            print(f"{col}: {unknown_count:.2%}")
        # Remove rows with 'unknown'
        data = data[data[col] != 'unknown']
        
    # Calculate rows removed
    rows_removed = total_rows - len(data)
        
    # Reset the index of the resulting DataFrame
    print(f'Rows with "unknown" values have been removed: {rows_removed} rows.')
    print()
    return data.reset_index(drop=True)

# A function to drop columns
def drop_columns(data, columns):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.
        columns:             Single or list of columns to drop.

    Purpose:
    Drop columns within DataFrame. 
    """
    data = data.drop(columns, axis=1)
    return data

# A function to calculate and generate table with outlier statistics 
def summarize_outliers(data, numeric_columns):
    """
    Arguments:
        data (pd.DataFrame):    The DataFrame containing the data.
        numeric_columns (list): List of numeric column names to analyze for outliers.

    Purpose:
    Generate a summary DataFrame of outlier statistics for numeric columns 
    in order to conduct data imputation.

    Returns:
    A summary DataFrame containing column name, lower limit, upper limit, 
    interquantile range (IQR) and total number of outliers within the numeric column.
    """
    # Create an empty list to store the results
    outlier_summary = []

    # Check by numeric columns
    for column_name in numeric_columns:
        # Calculate percentiles
        percentile25 = data[column_name].quantile(0.25)
        percentile75 = data[column_name].quantile(0.75)
        # Calculate interquartile range
        iqr = percentile75 - percentile25
        # Calculate upper and lower thresholds for outliers
        lower_limit = percentile25 - (iqr * 1.5)
        upper_limit = percentile75 + (iqr * 1.5)
        # Filter the outliers
        outliers = data[(data[column_name] > upper_limit) | (data[column_name] < lower_limit)]
        # Append the outlier statistics to the summary list
        outlier_summary.append({
            'column': column_name,
            'lower_limit': round(lower_limit, 2),
            'upper_limit': round(upper_limit, 2),
            'IQR': round(iqr, 2),
            'num_outliers': len(outliers),
        })
    # Return the summary list as DataFrame
    return pd.DataFrame(outlier_summary)

# A function to impute outliers with upper limit for the specified column
def impute_outliers_with_upper_limit(data, column_name, upper_limit):
    """
    Arguments:
    data (pd.DataFrame): The DataFrame containing the data.
    column_name (str):   The name of the column to impute outliers for.
    upper_limit (float): The upper limit value to replace outliers with.

    Purpose:
    Impute outliers in a specified column with the upper limit value.

    Returns:
    Outliers within the specified column imputed.
    """
    # Replace outliers with the upper limit
    data.loc[data[column_name] > upper_limit, column_name] = upper_limit

    # Print summary
    print(f'Imputed outliers in "{column_name}" with Upper Limit:')
    print(f'Upper limit: {upper_limit}')
    print(f'Maximum value after imputation: {data[column_name].max()}')
    print()
    return data

# A function to calculate the Variance Inflation Factor (VIF) for continuous independent variables.
def calculate_vif(data, columns):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.
        columns (list):      A list of column names representing the continuous independent variables.

    Purpose:
    Calculate the Variance Inflation Factor (VIF) for a set of continuous independent variables.

    Returns:
        pd.DataFrame: A DataFrame containing the VIF values for the specified columns.
    """
    # Subset of data with the specified continuous independent variables
    X_vif = data[columns]

    # Calculate the variance inflation factor for each variable
    vif = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    # Return the VIF results as a DataFrame
    return pd.DataFrame(vif, index=X_vif.columns, columns=['VIF'])

## Step 4: Data Visualization
# A function to plot a box plot for a column 
def plot_column_boxplot(data, column_name, title_suffix="BEFORE impute outliers"):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str):   The name of the column to plot.
        title_suffix (str):  A suffix to append to the plot title.
    
    Purpose: 
    Plot a box plot for a specified column in the dataset.
    
    Returns:
    Displays the box plot.
    """
    # Create the box plot
    sns.boxplot(data=data, x=column_name)
    plt.title(f'Box Plot for {column_name} {title_suffix}')
    plt.xlabel(column_name)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.show()

# A function to plot SINGLE barplot for grouped data where the target column values are aggregated by their SUM.
def plot_barplot_groupby_sum(data, group_col, target_col, xlabels=None, ylabels=None, title=None):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.
        group_col (str):     The column name in `data` to group by.
        target_col (str):    The column name in `data` whose values will be summed for each group.
        xlabels (optional):  Label for the x-axis. Defaults to None (without label).
        ylabels (optional):  Label for the y-axis. Defaults to None (without label).
        title (optional):    Title for the barplot. Defaults to None (no title).

    Purpose:
    Create a SINGLE barplot for grouped data, where the target column values are aggregated by their SUM.

    Returns:
    Displays SINGLE barplot.
    """
    group_data = data.groupby(group_col, observed=False)[target_col].sum()
    sns.barplot(x=group_data.index, y=group_data.values, hue=group_data.index)
    if xlabels:
        plt.xlabel(xlabels, fontsize=LABEL_FONT)
    if ylabels:
        plt.ylabel(ylabels, fontsize=LABEL_FONT)
    if title:
        plt.title(title, fontsize=TITLE_FONT)
    plt.show();

# A function to plot 2 barplot for grouped data where the target column values are aggregated by their SUM
def plot_2barplot_groupby_sum(data, group_col, target_col, ax, xlabels=None, title=None):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.
        group_col (str): The column name in `data` to group by.
        target_col (str): The column name in `data` whose values will be summed for each group.
        ax : To specify Matplotlib Axes object (ax[0] and ax[1])
        xlabels (optional): Label for the x-axis. Defaults to None (without label).
        title (optional): Title for the barplot. Defaults to None (no title).

    Purpose:
    Create 2 barplot for grouped data, where the target column values are aggregated by their SUM.

    Returns:
    Displays 2 barplot.
    """
    group_data = data.groupby(group_col, observed=False)[target_col].sum()
    sns.barplot(x=group_data.index, y=group_data.values, ax=ax)
    if xlabels:
        ax.set_xticks(range(len(group_data.index)))
        ax.set_xticklabels(xlabels, fontsize=LABEL_FONT)
    if title:
        ax.set_title(title, fontsize=TITLE_FONT)
    return ax

# A function to plot 2 barplot for grouped data where the target column values are aggregated by their MEAN value
def plot_2barplot_groupby_mean(data, group_col, target_col, ax, xlabels=None, title=None):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.
        group_col (str): The column name in `data` to group by.
        target_col (str): The column name in `data` whose values will be summed for each group.
        ax : To specify Matplotlib Axes object (ax[0] and ax[1])
        xlabels (optional): Label for the x-axis. Defaults to None (without label).
        title (optional): Title for the barplot. Defaults to None (no title).

    Purpose:
    Create 2 barplot for grouped data, where the target column values are aggregated by their MEAN value.

    Returns:
    Displays 2 barplot.
    """
    group_data = data.groupby(group_col, observed=False)[target_col].mean()
    sns.barplot(x=group_data.index, y=group_data.values, ax=ax)
    if xlabels:
        ax.set_xticks(range(len(group_data.index)))
        ax.set_xticklabels(xlabels, fontsize=LABEL_FONT)
    if title:
        ax.set_title(title, fontsize=TITLE_FONT)
    return ax

## Step 6: Modelling
# A function to generate a table with model's validation results (with GridSearchCV) 
def make_validation_results(model_name: str, best_model, X_val_scaled, y_val):
    """
    Arguments:
        model_name (string): Model name to display in the results table.
        best_model:          Best estimator from GridSearchCV.
        X_val_scaled:        Validation features (scaled).
        y_val:               Validation labels.

    Purpose:
    Returns a pandas dataframe with the precision, recall, F1, accuracy, and AUC 
    scores for the best model applied to the validation dataset.

    Returns:
    A table containing the validation metrics.
    """
    # Generate predictions and probabilities on the validation dataset
    val_predictions = best_model.predict(X_val_scaled)
    val_probabilities = best_model.predict_proba(X_val_scaled)[:, 1]  # For AUC

    # Calculate metrics
    precision = precision_score(y_val, val_predictions)
    recall = recall_score(y_val, val_predictions)
    f1 = f1_score(y_val, val_predictions)
    accuracy = accuracy_score(y_val, val_predictions)
    auc = roc_auc_score(y_val, val_probabilities)

    # Create a table of results
    val_results_table = pd.DataFrame({'model': [model_name],
                                      'precision': [precision],
                                      'recall': [recall],
                                      'F1': [f1],
                                      'accuracy': [accuracy],
                                      'auc': [auc]})

    return val_results_table

# A function to generate a table with model test results
def get_test_scores(model_name:str, y_pred, y_test_data):
    '''
    Arguments: 
        model_name (string):  Model name that will be show in the output table
        model:                A model
        y_pred:               y prediction with X_test data
        y_test_data:          Numpy array of y_test data
    
    Purpose: 
    Generate a table of test scores.

    Returns: 
    DataFrame with precision, recall, f1, accuracy, and AUC scores of the model.
    '''
    # Evaluating the scores with prediction values and actual values from test data
    auc = roc_auc_score(y_test_data, y_pred)
    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    recall = recall_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)

    # Return the results in DataFrame
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
    return table
