import pandas as pd
import numpy as np
import mlflow

from environment import hprint, print
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold


def training(X_training: pd.DataFrame,
             X_test: pd.DataFrame,
             y_training: pd.Series,
             y_test: pd.Series) -> Union[LogisticRegression,
                                         DecisionTreeClassifier,
                                         SVC, RandomForestClassifier]:
    best_model, score = None, 0
    # Documenting the experiment
    mlflow.set_experiment("Heart Disease Classification")

    for model in [LogisticRegression(max_iter=200),
                  DecisionTreeClassifier(max_depth=5),
                  SVC(kernel='linear'),
                  RandomForestClassifier(n_jobs=-1, class_weight='balanced',
                                         max_depth=5)]:
        model_name = type(model).__name__

        with mlflow.start_run(run_name=model_name):
            hprint(f'Training a {model} model...')
            run_id = mlflow.active_run().info.run_id
            print(f'Run ID: {run_id}')

            # Training the model
            model.fit(X_training, y_training)

            # Predicting
            y_pred = model.predict(X_test)

            # Evaluating the model
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_val_score(model, X_training, y_training,
                                         cv=kfold,
                                         scoring='balanced_accuracy')
            print(f'Mean Balanced Accuracy: {np.mean(cv_results)}')
            print(f'95% CI: {np.quantile(cv_results, [0.025, 0.975])}')

            bal_accuracy = balanced_accuracy_score(y_test, y_pred)
            if bal_accuracy > score:
                score = bal_accuracy
                best_model = model
            print(f'Balanced Accuracy in testing: {bal_accuracy}')
            cm_result = confusion_matrix(y_test, y_pred)
            print('Confusion matrix:', cm_result)
            tn, fp, fn, tp = cm_result.ravel()
            print(f'tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}')

            # Documenting the run
            mlflow.sklearn.log_model(model, model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric('BalancedAccuracy', bal_accuracy)
    return best_model
