# Data manipulation
import pandas as pd

# Functions 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import helper functions
from helpers import load_data, check_class_proportion, drop_duplicates, rename_column, check_and_remove_unknown,\
                    drop_columns, summarize_outliers, impute_outliers_with_upper_limit,\
                    calculate_vif

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# Constants within preprocessing.py
AGE_LIST = ['16-25','26-35' ,'36-45','46-55','56-65','66-75','76-85','86-95']
AGE_BINS = [16,25,35,45,55,65,75,85,95]
RANDOM_STATE = 42           # For reproducibility
TEST_SIZE = 0.25            # Split ratio for training and test data

## Step 1: Load the dataset
file_path = r"C:\Users\cando\OneDrive\Desktop\Workspaces\telemarking\Portuguese_telemarketing.csv"
df = load_data(file_path)
 
## Step 2: Inspect the data
def inspect_data(df):
    # Inspect the first 5 rows and last 5 rows of the dataset
    def first_last_rows(df):
        """
        Arguments:
            data (pd.DataFrame): The DataFrame containing the data.

        Purpose:
        Inspect the dataset and its values.
        """
        print('First 5 rows of the dataset:')
        print(df.head())
        print()
        print('Last 5 rows of the dataset:')
        print(df.tail())
        print()
    
    first_last_rows(df)

    # Inspect the information of the dataset including checking for null values, duplicated rows and so on. 
    def dataset_info(df):
        """
        Arguments:
            data (pd.DataFrame): The DataFrame containing the data.

        Purpose:
        1. Show the dataset information, null values and duplicated rows within the dataset. 
        2. Inspect the proportion of new customers within 'previous', 'poutcome' and 'pdays'. 
        """
        print(df.info())
        print()
        print(df.describe())
        print()
        print('Null values within dataset:')
        print(df.isna().sum())
        print()
        print(f'Duplicated rows: {df.duplicated().sum()}')
        print()
    
    dataset_info(df)
    
    # Check the balance of the target class (minority class vs majority class)
    check_class_proportion(df, 'term_deposit')

    # Inspect the proportion of new customers within 'previous', 'poutcome' and 'pdays' columns
    # 'previous' == 0 refers to 0 contacts performed before the campaign
    print(f'New customers within "previous" column: {(df["previous"] == 0).sum()} out of {df.shape[0]}')
    print()

    # 'poutcome' == 'nonexistent' refers to customers didn't participate the campaign before
    print(f'New customers within "poutcome" column: {(df["poutcome"] == 'nonexistent').sum()} out of {df.shape[0]}')
    print()

    # 'pdays' == '999' refers to customer was not contacted in previous campaign
    print(f'New customers within "pdays" column: {(df["pdays"] == 999).sum()} out of {df.shape[0]}')
    print() 

    # Return the DataFrame
    return df

## Step 3: Data Cleaning 
def clean_data(df):
    """
    Arguments:
        data (pd.DataFrame): The DataFrame containing the data.

    Purpose:
    1. Remove rows with 'unknown' values in object-type columns.
    2. Drop some columns.
    3. Inspect and impute outliers.
    4. Converting age into age groups and adding sequences to month and education.
    5. Inspect VIF for highly correlated variables and remove variable. 
    
    Returns:
    A new DataFrame with rows containing 'unknown' removed.
    """
    # Remove duplicated rows
    df = drop_duplicates(df)

    # Rename column names by 
    df = rename_column(df)

    # Check the 'unknown' proportion within the dataset, remove it by rows and return it as new df1
    # Reason: 'unknown' values are same as null values
    df1 = check_and_remove_unknown(df)

    # Drop 'previous', 'poutcome' and 'pdays' columns due to large proportion of new customers within these columns.
    # Reason: I would not able to examine will the customers participate the campaign if they have previous experiences on the last campaign.  
    df1 = drop_columns(df1, ['previous','poutcome', 'pdays'])
    
    # Drop 'duration' column 
    # Reason: 'duration' can only be collect after the campaign which is not realistic to use it for prediction for upcoming campaign
    # And this column can be highly affects the output target
    df1 = drop_columns(df1, ['duration'])

    ## Check for outliers
    # Create a list of numeric columns within df1
    num_col = ['campaign', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']
    # Fit the helper function with list of numeric columns and df1 to summarize the outliers within columns
    outlier_summary = summarize_outliers(df1, num_col)
    # Print the outlier results 
    print('Outlier Statistics:')
    print(outlier_summary)
    print()

    ## Impute outliers
    # Impute the outliers within 'campaign' column with the value of upper limit (75th percentile + IQR * 1.5) 
    threshold_campaign = outlier_summary.loc[outlier_summary['column'] == 'campaign', 'upper_limit'].values[0]
    impute_outliers_with_upper_limit(df1, 'campaign', threshold_campaign)

    # Impute the outliers within 'cons_conf_idx' column with the value of upper limit (75th percentile + IQR * 1.5) 
    threshold_cons_conf_idx = outlier_summary.loc[outlier_summary['column'] == 'cons_conf_idx', 'upper_limit'].values[0]
    impute_outliers_with_upper_limit(df1, 'cons_conf_idx', threshold_cons_conf_idx)

    ## Changing 'age' column into different age group
    # Inspect min and max age
    print(f'Maximum age of customer: {df1["age"].max()}')
    print(f'Minimum age of customer: {df1["age"].min()}')
    print()
    # Create a list of age group and fit the age into the age group
    df1['age_group'] = pd.cut(df1['age'], bins=AGE_BINS, labels=AGE_LIST, right=True)

    ## Create month order and 'month' column and sort the column with month order
    month_order = ['jan','feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    df1['month'] = pd.Categorical(df1['month'], categories=month_order, ordered=True) 

    ## Create education level and sort 'education' column by education level
    education_level = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']
    df1['education'] = pd.Categorical(df1['education'], categories=education_level, ordered=True) 

    ## Drop 'age' column
    # Reason: Keeping 'age_group' column which is correlated with 'age' column
    df1 = drop_columns(df1, ['age'])

    ## Check for multicollinearity between variables
    high_corr_col = df1[['term_deposit','emp_var_rate', 'euribor3m', 'nr_employed']].corr()
    print('Correlation between Outcome Variable and Highly Correlated Predictors:')
    print(high_corr_col)
    print()

    # Calculate the variance inflation factor to check how much correlation between independent variables
    # Create a subset of the data with employment variation rate, euribor 3m rate and number of employees
    vif_col1 = ['emp_var_rate', 'euribor3m', 'nr_employed']
    # Fit the function to calculate the vif
    df1_vif_before = calculate_vif(df1, vif_col1)
    print(df1_vif_before)
    print()

    # Drop 'nr_employed' 
    # Reason: Due to its interpretability with term deposit and multicollinearity with other variables
    df1 = df1.drop(columns=['nr_employed'], axis=1)

    # Double check the vif for 'emp_var_rate', 'euribor3m'
    # Create a subset of the data without 'nr_employed'
    vif_col2 = ['emp_var_rate', 'euribor3m']
    # Fit the function to calculate the vif
    df1_vif_after = calculate_vif(df1, vif_col2)
    print(df1_vif_after)
    print()

    ## Inspect number of rows and columns before encoding
    print(f'Number of rows and columns before encoding: {df1.shape}')
    print()
    print("Remaining columns before encoding:")
    print(df1.columns)
    print()

    return df1

## Step 5:Encode and modelling setup after data visualization
def encode_modelling_setup(df1):
    """
    Arguments:
        df1 (pd.DataFrame): The DataFrame containing the data.

    Purpose:
    1. Split the data (before encoding to ensure test data remain unseen for the models).
    2. Encode test and training data for modelling.
    
    Returns:
    Encoded X_test and X_train for modelling.
    """
    # Create X and y variable
    X = df1.drop('term_deposit', axis=1)
    y = df1['term_deposit']

    # Reset the indices to ensure the indices are in correct order
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # Double check X and y
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print()

    # Split the data into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # Encode the training and test sets
    def encode_data(X_train, X_test):
        # Drop 'month' and 'day_of_week' columns
        # Reason: These columns are not predictive columns which can only be use for data visualization
        drop_cols = ['day_of_week', 'month']
        X_train = X_train.drop(columns=drop_cols, axis=1)
        X_test = X_test.drop(columns=drop_cols, axis=1)

        # Encode binary columns ('default', 'loan', 'housing', 'contact')
        binary_col = ['default', 'loan', 'housing', 'contact']
        for col in binary_col:
            X_train[col] = X_train[col].astype('category').cat.codes
            X_test[col] = X_test[col].astype('category').cat.codes

        # Encode `education` and `age_group`
        for col in ['education', 'age_group']:
            X_train[col] = X_train[col].astype('category').cat.codes
            X_test[col] = X_test[col].astype('category').cat.codes

        # One-Hot Encode `job` and `marital` columns
        categorical_cols = ['job', 'marital']
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

        # Reindex test set to match training set columns
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Change boolean values into numeric values
        train_bool_cols = X_train.select_dtypes(include='bool').columns
        test_bool_cols = X_test.select_dtypes(include='bool').columns

        X_train[train_bool_cols] = X_train[train_bool_cols].astype(int)
        X_test[test_bool_cols] = X_test[test_bool_cols].astype(int)

        ## Final confirmation
        print(f'X_train shape after encoding: {X_train.shape}')
        print(f'X_test shape after encoding: {X_test.shape}')
        print()

        return X_train, X_test

    # Encode the training and test sets
    X_train, X_test = encode_data(X_train, X_test)

    # Split the encoded training set into training and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=TEST_SIZE, stratify=y_train, random_state=RANDOM_STATE)

    # Scale the training (X_tr) and validation (X_val) sets
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)  # Fit and transform on X_tr
    X_val_scaled = scaler.transform(X_val)    # Transform X_val

    # Return the required variables
    return X_train, X_tr_scaled, X_val_scaled, X_test, y_tr, y_test, y_train, y_val
