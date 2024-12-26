import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

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
    #print(model.summary())
    
    return model

def run_correlation_matrix(depression_df):
    # Compute the correlation matrix
    correlation_matrix = depression_df.corr()

    # Melt the correlation matrix into a long format
    correlation_data = correlation_matrix.reset_index().melt(id_vars='index', var_name='Predictor_2', value_name='Correlation')
    correlation_data.rename(columns={'index': 'Predictor_1'}, inplace=True)

    # Save to CSV for Power BI
    correlation_data.to_csv('Correlation_Data.csv', index=False)

def run_logistic_regression(depression_df):
    # Define predictors (X) and target (y)
    X = depression_df.drop(columns=['Depression_Yes'])
    y = depression_df['Depression_Yes']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    # Make predictions
    y_pred = log_model.predict(X_test)
    y_prob = log_model.predict_proba(X_test)[:, 1]  # Probability of being in class '1'

    # Evaluate the model
    #print("Logistic Regression Results:")
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    #print("Classification Report:\n", classification_report(y_test, y_pred))

    # Add actual values, predictions, and probabilities to X_test for display
    #X_test['Actual_Depression'] = y_test.values  # Add actual depression values for comparison
    #X_test['Predicted_Depression'] = y_pred
    #X_test['Probability'] = y_prob

    # Display all rows of predictions and probabilities
    #print("\nFull Predictions and Probabilities:")
    #print(X_test[['Actual_Depression', 'Predicted_Depression', 'Probability']].to_string())

    # Return the trained model for later use in manual predictions
    return log_model


def predict_depression(manual_input, model):
    # Ensure the input is in the same format as the training data
    input_data = pd.DataFrame([manual_input], columns=model.feature_names_in_)
    
    # Predict using the trained model
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    
    # Display prediction and probability
    if prediction == 1:
        print("The model predicts depression: Yes")
    else:
        print("The model predicts depression: No")
        
    #print(f"Prediction probability: {probability[0]}")



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

    # Run Logistic regression predictions
    logistic = run_logistic_regression(depression_df)

    # Predict using the trained model
    #           | |
    #           | |
    #           V V
    # Input features for depression prediction:
    # 1. Age (e.g., 18, 21, 25)
    # 2. Academic Pressure (1-5 scale: 1=low, 5=high)
    # 3. Study Satisfaction (1-5 scale: 1=low, 5=high)
    # 4. Study Hours (e.g., 2, 4, 6)
    # 5. Financial Stress (1-5 scale: 1=low, 5=high)
    # 6. Gender_Male (e.g., 1 for Male, 0 for Female)
    # 7. Sleep Duration_7-8 hours (e.g., 1 if '7-8 hours', otherwise 0)
    # 8. Sleep Duration_Less than 5 hours (e.g., 1 if 'Less than 5 hours', otherwise 0)
    # 9. Sleep Duration_More than 8 hours (e.g., 1 if 'More than 8 hours', otherwise 0)
    # 10. Dietary Habits_Moderate (e.g., 1 if 'Moderate', otherwise 0)
    # 11. Dietary Habits_Unhealthy (e.g., 1 if 'Unhealthy', otherwise 0)
    # 12. Suicidal Thoughts (e.g., 1 for 'Yes', 0 for 'No')
    # 13. Family History of Mental Illness (e.g., 1 for 'Yes', 0 for 'No')

    # Example input format: [22, 4, 3, 6, 4, 1, 1, 0, 0, 0, 1, 1, 0]
    # This represents: [Age, Academic Pressure, Study Satisfaction, Study Hours, Financial Stress,
    #                   Gender_Male, Sleep Duration_7-8 hours, Sleep Duration_Less than 5 hours,
    #                   Sleep Duration_More than 8 hours, Dietary Habits_Moderate, Dietary Habits_Unhealthy,
    #                   Suicidal Thoughts, Family History of Mental Illness]

    input = [28, 2.0, 4.0, 9, 2, 1, 1, 0, 0, 1, 0, 1, 1]

    logistic_model = run_logistic_regression(depression_df)
    predict_depression(input, logistic_model)
