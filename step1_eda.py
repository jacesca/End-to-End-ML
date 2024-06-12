"""
End-to-end project: Predicting heart disease
Goal: inform decision-making of cardiologists.

STEP1 - EDA
"""
import pandas as pd

from environment import (histogram_boxplot, labeled_barplot,
                         print, hprint)


def eda(df: pd.DataFrame,
        cols_for_hist: list, cols_for_count: list) -> pd.DataFrame:
    # Reading the data
    hprint('Exploratory data analysis')
    print('Head:', df.head())
    print('Column info:', df.info())
    print('Description:', df.describe().T)
    print('Null Values:', df.isnull().sum())

    # Reviewing the data distribution
    for col in cols_for_hist:
        histogram_boxplot(df[col])

    for col in cols_for_count:
        labeled_barplot(df[col])
        print(col, df[col].value_counts())
