import pandas as pd
from environment import (impute_missing_values,
                         impute_missing_values_with_KNN,
                         print, hprint)


def clean_data(df: pd.DataFrame,
               columns_to_simpleimpute: list = []) -> pd.DataFrame:
    hprint('Cleaning data')

    # Drop empty columns
    df_clean = df.drop('oldpeak', axis='columns')  # axis=1

    # Drop duplicate rows
    df_clean = df_clean.drop_duplicates()

    # Impute missing values
    df_clean, _ = impute_missing_values(df_clean, columns_to_simpleimpute)
    df_clean[df_clean.columns], _ = impute_missing_values_with_KNN(df_clean)
    print(df_clean.isnull().sum())

    return df_clean
