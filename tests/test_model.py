import unittest
import pandas as pd
import numpy as np
import mlflow

from environment import (SAVED_MODEL_PATH, TEST_DATA_PATH,
                         FEATURES_SELECTED_PATH,
                         SCALER_PATH,
                         load_object, print)
from sklearn.metrics import balanced_accuracy_score


# Create a class called TestModelInference
class TestModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.read_csv(TEST_DATA_PATH, index_col=0)
        features_selected = load_object(FEATURES_SELECTED_PATH)
        scaler = load_object(SCALER_PATH)
        target_col = 'target'

        cls.X_test = df.drop(target_col, axis='columns')
        cls.X_test = pd.DataFrame(data=scaler.transform(cls.X_test),
                                  columns=cls.X_test.columns)
        cls.X_test = cls.X_test[features_selected]
        cls.y_test = df[target_col]

        print('Features:', features_selected)
        print('Data head:', cls.X_test.head())

        cls.model = mlflow.sklearn.load_model(SAVED_MODEL_PATH)
        cls.y_pred = cls.model.predict(cls.X_test)

    def test_prediction_output_shape(self):
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(y_pred.shape[0], self.X_test.shape[0],
                         "Shape of prediction is not what we expected!")

    def test_prediction_output_values(self):
        unique_values = np.unique(self.y_pred)
        self.assertEqual(set(unique_values), {0, 1},
                         "Predicted values are out of range!")

    def test_prediction_accuracy(self):
        accuracy = balanced_accuracy_score(self.y_test, self.y_pred)
        threshold = 0.85
        self.assertGreaterEqual(
            accuracy, threshold,
            f"Accuracy ({accuracy}) is below the threshold ({threshold})!"
        )


if __name__ == "__main__":
    unittest.main()
