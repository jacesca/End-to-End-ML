import pandas as pd
import logging
import mlflow

from environment import (prepare_environment, end_environment,
                         hprint, print, SAVED_MODEL_PATH,
                         LOGS_PATH)
from step2_cleaning import clean_data
from step3_feature_eng import feature_engineering


# Starting the environment
hprint('Preparing the environment')
prepare_environment()

# Reading the data
hprint('Reading the data')
df = pd.read_csv('datasets/heart_disease_df_1.csv')
target_col = 'target'

# Cleaning the data
df_clean = clean_data(df, columns_to_simpleimpute=['restecg'])

# Making some feature engineering
X_train, X_test, y_train, y_test, standardizer, selector = \
    feature_engineering(df_clean, target_col, show_graph=False)

# Loading the model
model = mlflow.sklearn.load_model(SAVED_MODEL_PATH)
print('Selected Model:', model)

# Setting up basic logging configuration
hprint('Setting up the logging')
logging.basicConfig(filename=LOGS_PATH, level=logging.INFO)

# Make predictions on the test set and log the results
for i in range(X_test.shape[0]):
    instance = X_test.iloc[i].to_frame().T
    prediction = model.predict(instance)  # ex. [0.]
    logging.info(
        f'Instance {i} - '
        f'PredClass: {prediction[0]}, '
        f'RealClass: {y_test.iloc[i]}'
    )

# Reading the logs
hprint('Reading the logs')
with open(LOGS_PATH, 'r') as f:
    lines = f.readlines()
    predicted_class = [int(float(line.split('PredClass: ')[1].split(',')[0]))
                       for line in lines]
print('Predicted class Sample:', predicted_class[:5])

# Ending
end_environment()
