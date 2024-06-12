import pandas as pd
import mlflow
import os
import shutil
import glob

from environment import (prepare_environment, end_environment,
                         hprint, print, SAVED_MODEL_PATH,
                         SAVED_FIGURE_PATH)
from step1_eda import eda
from step2_cleaning import clean_data
from step3_feature_eng import feature_engineering
from step4_training import training


# Starting the environment
hprint('Preparing the environment')
prepare_environment()
for dir_name in [SAVED_MODEL_PATH, 'mlruns']:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
files = glob.glob(f'{SAVED_FIGURE_PATH}*.png')
for file in files:
    os.remove(file)

# Reading the data
hprint('Reading the data')
df = pd.read_csv('datasets/heart_disease_df_1.csv')
cols_for_hist = ['age', 'trestbps', 'chol', 'thalach']
cols_for_count = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope',
                  'ca', 'thal', 'target']
target_col = 'target'

# Making an exploratory data analysis
eda(df, cols_for_hist, cols_for_count)

# Cleaning the data
df_clean = clean_data(df, columns_to_simpleimpute=['restecg'])

# Making some feature engineering
X_train, X_test, y_train, y_test, standardizer, selector = \
    feature_engineering(df_clean, target_col)

# Training different models
model = training(X_train, X_test, y_train, y_test)
print('Selected Model:', model)

# Saving the selected model
mlflow.sklearn.save_model(model, f'{SAVED_MODEL_PATH}')

# Ending
end_environment()
