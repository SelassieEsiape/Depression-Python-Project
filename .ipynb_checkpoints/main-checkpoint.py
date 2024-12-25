import pandas as pd, numpy as np, statsmodels.api as sm
from tabulate import tabulate


data = pd.read_csv('Depression Student Dataset.csv')
depression_df = pd.DataFrame(data)


for col in depression_df:
    if not pd.api.types.is_numeric_dtype(depression_df[col]):
        depression_df = pd.get_dummies(depression_df, columns=[col], prefix=col, drop_first=True)

depression_df.head()
