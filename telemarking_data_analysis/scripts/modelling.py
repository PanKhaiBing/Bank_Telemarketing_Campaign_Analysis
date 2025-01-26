# Data manipulation
import pandas as pd

# Data modelling
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Metrics and helpful functions
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix                 
from sklearn.preprocessing import StandardScaler

# Import helper functions
from helpers import make_validation_results, get_test_scores 

# Constants 
RANDOM_STATE = 42           # For reproducibility
CV = 5                      # Number of k-fold cross validation
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
REFIT_METRIC = 'recall'     # Primary metric that the model will be focus on (minimizing False Negative, missed opportunities)

## Step 6: Modelling 
## First model -- Decision Tree model 
def decision_tree_model(X_tr_scaled, y_tr, X_val_scaled, y_val):

    # Instantiate the model    
    tree = DecisionTreeClassifier(class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE)

    # Assign a dictionary of hyperparameters to search over
    dt_cv_params = {'max_depth':[4, 6, 8, 12],
                    'min_samples_leaf': [2, 4, 6],
                    'min_samples_split': [2, 4, 6]
                    }

    # Instantiate GridSearch with hyperparameters
    dt_cv = GridSearchCV(tree, dt_cv_params, scoring=SCORING, cv=CV, refit=REFIT_METRIC)

    # Fit the training data set into GridSearch to find the best parameters through cv
    dt_cv.fit(X_tr_scaled, y_tr)
    
    # Assign the best parameters as best_dt
    best_dt = dt_cv.best_estimator_

    # Use the function to validate the model with validation set and generate a table of results 
    dt_val_results = make_validation_results('Decision Tree', best_dt, X_val_scaled, y_val)

    # Print the best parameters found with Decision Tree GridSearch
    print('Decision Tree best parameters:')
    print(dt_cv.best_params_)
    print()

    # Print Decision Tree validation results
    print('Decision Tree Validation Results Table:')
    print(dt_val_results)
    print()

    # Return the validation results table
    return dt_val_results

## Second model -- Random Forest model: round 1
def random_forest_model(X_tr_scaled, y_tr, X_val_scaled, y_val):

    # Instantiate the model
    rf = RandomForestClassifier(class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE)

    # Assign a dictionary of hyperparameters to search over
    rf_cv_params = {'max_depth': [3, 5, 10, 15], 
                    'max_features': ["sqrt"],
                    'max_samples': [0.3, 0.5, 0.7, 0.9],
                    'min_samples_leaf': [2, 4, 6],
                    'min_samples_split': [3, 7, 10],
                    'n_estimators': [300, 500],
                    }  

    # Instantiate GridSearch with hyperparameters
    rf_cv = GridSearchCV(rf, rf_cv_params, cv=CV, scoring=SCORING, refit=REFIT_METRIC, n_jobs=N_JOBS, verbose=VERBOSE)

    # Fit the training data set into GridSearch to find the best parameters through cv
    rf_cv.fit(X_tr_scaled, y_tr)

    # Assign the best parameters as best_rf
    best_rf = rf_cv.best_estimator_

    # Use the function to validate the model with validation set and generate a table of results 
    rf_val_results = make_validation_results('Random Forest', best_rf, X_val_scaled, y_val)

    # Print the best parameters found with Random Forest GridSearch
    print('Random Forest best parameters:')
    print(rf_cv.best_params_)
    print()

    # Print Random Forest validation results
    print('Random Forest Validation Results Table:')
    print(rf_val_results)
    print()

    # Return the validation results table
    return rf_val_results, best_rf
    
## Third model -- XGBoost
def xgboost_model(X_tr_scaled, y_tr, X_val_scaled, y_val):

    # Instantiate the model    
    xgb = XGBClassifier(objective='binary:logistic', scale_pos_weight=len(y_tr[y_tr == 0]) / len(y_tr[y_tr == 1]), random_state=RANDOM_STATE)

    # Assign a dictionary of hyperparameters to search over
    xgb_cv_params = {'max_depth': [4, 6],
                     'min_child_weight': [3, 5],
                     'learning_rate': [0.1, 0.2, 0.3],
                     'n_estimators': [5, 10, 15],
                     'subsample': [0.7],
                     'colsample_bytree': [0.7]
                    }

    # Instantiate GridSearch with hyperparameters
    xgb_cv = GridSearchCV(xgb, xgb_cv_params, scoring=SCORING, cv=CV, verbose=VERBOSE, n_jobs=N_JOBS, refit=REFIT_METRIC)

    # Fit the training data set into GridSearch to find the best parameters
    xgb_cv.fit(X_tr_scaled, y_tr)

    # Assign the best parameters as best_xgb
    best_xgb = xgb_cv.best_estimator_

    # Use the function to validate the model with validation set and generate a table of results 
    xgb_val_results = make_validation_results('XGBoost', best_xgb, X_val_scaled, y_val)

    # Print the best parameters found with XGBoost GridSearch
    print('XGBoost best parameters:')
    print(xgb_cv.best_params_)
    print()

    # Print XGBoost validation results
    print('XGBoost Validation Results Table:')
    print(xgb_val_results)
    print()

    # Return the validation results table
    return xgb_val_results

## Fourth model -- Logistic regression model (with no GridSearch)
def logistic_regression_model(X_tr_scaled, y_tr, X_val_scaled, y_val):

    # Instantiate the model
    log_clf = LogisticRegression(class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE, max_iter=1000)

    # Instantiate GridSearch
    log_reg_cv = GridSearchCV(log_clf, param_grid={}, scoring=SCORING, cv=CV, verbose=VERBOSE, n_jobs=N_JOBS, refit=REFIT_METRIC)

    # Fit the training data set into GridSearch 
    log_reg_cv.fit(X_tr_scaled, y_tr)
    
    # Assign the best parameters as best_log_reg but there will be no hyperparameters and GridSearch 
    best_log_reg = log_reg_cv.best_estimator_

    # Use the function to validate the model with validation set and generate a table of results 
    log_reg_results = make_validation_results('Logistic Regression', best_log_reg, X_val_scaled, y_val)

    # Print Logistic Regression validation results
    print('Logistic Regression Validation Results Table:')
    print(log_reg_results)
    print()

    # Return the validation results table
    return log_reg_results

## Random Forest model: round 2 -- train on whole training data set and test data
def random_forest_model_test_data(X_train, y_train, X_test, y_test, best_rf):

    # Rescale the X_train and X_test 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the entire training data
    X_test_scaled = scaler.transform(X_test)        # Transform the test data using the same scaler

    # Retrain the model on the entire training set with best hyperparameters found in GridSearch
    best_rf.fit(X_train_scaled, y_train)

    # Test the model with test data
    y_pred = best_rf.predict(X_test_scaled)

    # Fit the function to display model test scores
    rf2_test_scores = get_test_scores('random forest2 test', y_pred, y_test)

    # Print the test results
    print('Random Forest on Test Data:')
    print(rf2_test_scores)
    print()

    ## Confusion matrix 
    # Generate array of values for confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)

    ## Gini importance
    rf2_importances = pd.DataFrame(best_rf.feature_importances_, 
                                    columns=['gini_importance'], 
                                    index=X_test.columns
                                    )
    # Sort gini importance from the highest to lowest
    rf2_importances = rf2_importances.sort_values(by='gini_importance', ascending=False)

    # Only extract the features with importances > 0
    rf2_importances = rf2_importances[rf2_importances['gini_importance'] != 0]
    print(rf2_importances)

    # Return the test results
    return best_rf, rf2_test_scores, cm, rf2_importances


