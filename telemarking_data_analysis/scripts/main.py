# Data manipulation
import pandas as pd

# Import helper functions
from helpers import load_data
                    
# Import functions from preprocessing script
from preprocessing import inspect_data, clean_data, encode_modelling_setup

# Import functions from visualization script
from visualization import data_visualization, plot_findings

# Import functions from modelling script
from modelling import decision_tree_model, random_forest_model, xgboost_model, \
                      logistic_regression_model,random_forest_model_test_data

# Constants within all the scripts
AGE_LIST = ['16-25','26-35' ,'36-45','46-55',
            '56-65','66-75','76-85','86-95'] # Age list for converting age into different age group
AGE_BINS = [16,25,35,45,55,65,75,85,95]
TITLE_FONT = 16             # Font size for title in graph
LABEL_FONT = 12             # Font size for label in graph
RANDOM_STATE = 42           # For reproducibility
TEST_SIZE = 0.25            # Split ratio for training and test data
FIGSIZE = [16,6]            # Size of the figure for consistency 
CLASS_WEIGHT = 'balanced'   # To counter imbalanced class within the target variable (87% vs 13%)
SCORING = {                 # Dictionary of scoring metrics 
    'accuracy': 'accuracy', 
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
} 
VERBOSE = 1                 # Verbosity level of GridSearchCV
N_JOBS = -1
REFIT_METRIC = 'recall'     # Primary metric that the model will be focus on (minimizing False Negative / missed opportunities)

# Step 1: Load the dataset
file_path = r"C:\Users\cando\OneDrive\Desktop\Workspaces\telemarking_data_analysis\data\Portuguese_telemarketing.csv"
df = load_data(file_path)

# Step 2: Inspect the data 
df = inspect_data(df)     

# Step 3: Data Cleaning
df1 = clean_data(df)

# Step 4: Data Visualization
data_visualization(df1)

# Step 5: Encode and prepare the data for modelling
X_train, X_tr_scaled, X_val_scaled, X_test, y_tr, y_test, y_train, y_val = encode_modelling_setup(df1)

# Step 6: Modelling 
# Decision Tree 
decision_tree_model(X_tr_scaled, y_tr, X_val_scaled, y_val)

# Random Forest
rf_val_results, best_rf = random_forest_model(X_tr_scaled, y_tr, X_val_scaled, y_val)

# XG Boost
xgboost_model(X_tr_scaled, y_tr, X_val_scaled, y_val)

# Logistic Regression
logistic_regression_model(X_tr_scaled, y_tr, X_val_scaled, y_val)

# Random Forest (champion model) on test data
best_rf, rf2_test_scores, cm, rf2_importances = random_forest_model_test_data(X_train, y_train, X_test, y_test, best_rf)

## Step 7: Visualize Random Forest test results
plot_findings(best_rf, cm, rf2_importances)

