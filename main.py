import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



def preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    depression_df = pd.DataFrame(data)
    
    # Convert categorical columns to dummy variables
    for col in depression_df:
        if not pd.api.types.is_numeric_dtype(depression_df[col]):
            depression_df = pd.get_dummies(depression_df, columns=[col], prefix=col, drop_first=True)
    # Ensure all columns are numeric (int or float)
    depression_df = depression_df.astype({col: 'int' for col in depression_df.select_dtypes('bool').columns})

    return depression_df

def run_regression(depression_df):
    # Define target variable (y) and predictors (X)
    y = depression_df['Depression_Yes']
    X = depression_df.drop(columns=['Depression_Yes']).assign(const=1)
    
    # Fit the multiple linear regression model
    model = sm.OLS(y, X).fit()
    
    # Display model summary
    print(model.summary())
    
    return model

def run_correlation_matrix(depression_df):
    # Compute the correlation matrix
    correlation_matrix = depression_df.corr()

    # Melt the correlation matrix into a long format
    correlation_data = correlation_matrix.reset_index().melt(id_vars='index', var_name='Predictor_2', value_name='Correlation')
    correlation_data.rename(columns={'index': 'Predictor_1'}, inplace=True)

    # Save to CSV for Power BI
    correlation_data.to_csv('Correlation_Data.csv', index=False)



# Main script
if __name__ == "__main__":
    # Filepath to dataset
    filepath = 'Depression Student Dataset.csv'
    
    # Preprocess the data
    depression_df = preprocess_data(filepath)
    
    # Run regression model
    model = run_regression(depression_df)

    # Run correlation matrix
    matrix = run_correlation_matrix(depression_df)

    # Save processed data for visuals
    # depression_df.to_csv('Preprocessed_Depression_Student_Dataset.csv', index=False)


    
