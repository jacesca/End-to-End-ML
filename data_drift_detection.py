import pandas as pd
from scipy.stats import ks_2samp

from environment import (prepare_environment, end_environment,
                         hprint, print, population_stability_index)
from step2_cleaning import clean_data


# Starting the environment
hprint('Preparing the environment')
prepare_environment()

# Reading samples data
hprint('Reading the data')
df_1 = pd.read_csv('datasets/heart_disease_df_1.csv')
df_2 = pd.read_csv('datasets/heart_disease_cleaned_2.csv')
target_col = 'target'

# Cleaning the data
df_1 = clean_data(df_1, columns_to_simpleimpute=['restecg'])

for col in df_1.drop(target_col, axis='columns').columns:
    # perform the KS-test - ensure input samples are numpy arrays
    hprint(f'Detecting data drift for {col}')
    # with ks_2samp
    test_statistic, p_value = ks_2samp(df_1[col], df_2[col])
    if p_value < 0.05:
        print(f"ks_2samp: p-value={p_value}",
              "Reject null hypothesis - data drift might be occuring")
    else:
        print(f"ks_2samp: p-value={p_value}",
              "Samples are likely to be from the same dataset")

    # perform the PSI test
    psi_value, explanation = population_stability_index(df_1[col], df_2[col],
                                                        feature_name=col)
    print(f"PSI: {psi_value}", explanation)

# Ending
end_environment()
