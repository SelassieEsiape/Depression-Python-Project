import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

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

# Main script
if __name__ == "__main__":
    # Filepath to dataset
    filepath = 'Depression Student Dataset.csv'
    
    # Preprocess the data
    depression_df = preprocess_data(filepath)
    
    # Run regression model
    model = run_regression(depression_df)

    
