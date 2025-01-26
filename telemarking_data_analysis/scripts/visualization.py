# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay        

# Import helper functions
from helpers import load_data, plot_column_boxplot, plot_barplot_groupby_sum, plot_2barplot_groupby_sum, plot_2barplot_groupby_mean

# Constants within visualization.py
FIGSIZE = [14,6]
TITLE_FONT = 16        
LABEL_FONT = 12             

# Load the dataset
file_path = r"C:\Users\cando\OneDrive\Desktop\Workspaces\telemarking\Portuguese_telemarketing.csv"
df = load_data(file_path)

## Step 4: Data Visualization
def data_visualization(df1):
    """
    Arguments:
        df1 (pd.DataFrame): The DataFrame containing the data.

    Purpose:
    Display the relationship between variables with visualization. 
    
    Returns:
    Plot 1 to 4:  4 box plots for displaying data imputation effect.
    Plot 5 to 14: 9 bar plots to display relationship between term deposit and variables.
    Plot 15:      Confusion matrix of Random Forest on test results.
    Plot 16:      Gini importance of Random Forest on test results.
    """
    ## Effect for Data Imputation with outliers
    def plot_imputation_effect(df, df1):
        # Plot 1 and 2: Box plot for 'campaign' column BEFORE and AFter impute outliers
        plot_column_boxplot(df, 'campaign', title_suffix="BEFORE impute outliers")
        plot_column_boxplot(df1, 'campaign', title_suffix="AFTER impute outliers")

        # Plot 3 and 4: Box plot for 'cons_conf_idx' column BEFORE and AFter impute outliers
        plot_column_boxplot(df, 'cons.conf.idx', title_suffix="BEFORE impute outliers")
        plot_column_boxplot(df1, 'cons_conf_idx', title_suffix="AFTER impute outliers")

    plot_imputation_effect(df, df1)
    
    # Plot 5: Term deposit bar plot
    def plot_term_deposit_barplot(df1):
        term_deposit_counts = df1['term_deposit'].value_counts().sort_index()  
        plt.figure(figsize=FIGSIZE)
        sns.barplot(x=term_deposit_counts.index, y=term_deposit_counts.values)
        plt.xlabel("Term Deposit", fontsize=LABEL_FONT)
        plt.xticks(ticks=[0, 1], labels=['Who are not', 'Who placed'], fontsize=10)
        plt.title("Customers who placed term deposit and who are not", fontsize=TITLE_FONT)
        plt.tight_layout()
        plt.show()

    plot_term_deposit_barplot(df1) 

    ## Plot 6: Term Deposit VS different JOBS bar plot
    def plot_term_deposit_vs_job(df1):
        plt.figure(figsize=FIGSIZE)
        plot_barplot_groupby_sum(df1, 'job', 'term_deposit', xlabels='Job', ylabels='Number of Term Deposit', title='Term Deposit by Job')

    plot_term_deposit_vs_job(df1)

    ## Plot 7: Term Deposit VS different AGE GROUP bar plot
    def plot_term_deposit_vs_age_group(df1):
        plt.figure(figsize=FIGSIZE)
        plot_barplot_groupby_sum(df1, 'age_group', 'term_deposit', xlabels='Age Group', ylabels='Number of Term Deposit', title='Term Deposit by Age Group')

    plot_term_deposit_vs_age_group(df1)

    ## Plot 8: Correlation Heatmap
    def correlation_heatmap(df1):
        plt.figure(figsize=FIGSIZE)
        heatmap = sns.heatmap(df1.corr(numeric_only=True), annot=True, vmin=-1, vmax=1, cmap=sns.color_palette("vlag", as_cmap=True))
        heatmap.set_title("Correlation Heatmap")
        plt.show()
    
    correlation_heatmap(df1)

    ## Plot 9: Monthly term deposit VS Monthly average of euribor 3 month rate 
    def plot_term_deposit_vs_average_euribor3m(df1):
        fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
        plot_2barplot_groupby_sum(df1, 'month', 'term_deposit',ax=ax[0], title='Monthly term deposit')
        plot_2barplot_groupby_mean(df1, 'month', 'euribor3m',ax=ax[1], title='Monthly average euribor rate')
        plt.show()
    
    plot_term_deposit_vs_average_euribor3m(df1)

    ## Plot 10 and 11: Inspect AMOUNT of TERM DEPOSIT on customers with or without HOUSING LOAN and customers with or without PERSONAL LOAN
    def plot_term_deposit_on_housing_loan_personal_loan(df1):
        fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
        plot_2barplot_groupby_sum(df1, 'housing', 'term_deposit',ax=ax[0], xlabels=['Who have', 'Who does not have'], title='Term Deposit by "Housing Loan"')
        plot_2barplot_groupby_sum(df1, 'loan', 'term_deposit',ax=ax[1], xlabels=['Who have', 'Who does not have'], title='Term Deposit by "Personal Loan"')
        plt.show()
    
    plot_term_deposit_on_housing_loan_personal_loan(df1)
    
    ## Plot 12 and 13: Inspect AMOUNT of TERM DEPOSIT based on customer marital status and contact communication type
    def plot_term_deposit_on_marital_contact(df1):
        fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
        plot_2barplot_groupby_sum(df1, 'marital', 'term_deposit',ax=ax[0], title='Term Deposit by Marital Status')
        plot_2barplot_groupby_sum(df1, 'contact', 'term_deposit',ax=ax[1], title='Term Deposit by Contact Type')
        plt.show()
    
    plot_term_deposit_on_marital_contact(df1)

    ## Plot 14: Education level VS Term Deposit
    def plot_term_deposit_vs_education(df1):
        plt.figure(figsize=FIGSIZE)
        plot_barplot_groupby_sum(df1, 'education', 'term_deposit', xlabels='Education level', ylabels='Number of Term Deposit', title='Term Deposit by Education level')

    plot_term_deposit_vs_education(df1)

## Step 7: Visualize Random Forest test results
def plot_findings(best_rf, cm, rf2_importances):
    """
    Arguments:
        best_rf :         The trained random forest model with best hyperparameters.
        cm :              The confusion matrix.
        rf2_importances : Random Forest feature importances in DataFrame format.
    
    Purpose:
    Plot the findings of the Random Forest model with test data.

    Returns:
    1. The confusion matrix.
    2. Random Forest feature importances based on Gini importance.
    """
    # Plot 15: Random Forest confusion matrix on test data results
    def plot_confusion_matrix(best_rf, cm):
        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=best_rf.classes_)
        disp.plot(values_format='d')
        plt.title("Confusion Matrix for Random Forest Model")
        plt.show()

    plot_confusion_matrix(best_rf, cm)

    # Plot 16: Feature importances using Gini importance
    def gini_importance(rf2_importances):
        sns.barplot(data=rf2_importances, x="gini_importance", y=rf2_importances.index, orient='h')
        plt.title("Random Forest: Feature Importances for Customers who place Term Deposit", fontsize=12)
        plt.ylabel("Feature")
        plt.xlabel("Importance")
        plt.show()

    gini_importance(rf2_importances)





