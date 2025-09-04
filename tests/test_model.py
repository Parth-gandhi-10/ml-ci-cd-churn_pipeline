import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier

class TestChurnModelTraining(unittest.TestCase):
    def test_model_training(self):
        # Load the saved churn model
        model = joblib.load("model/churn_model.pkl")

        # Check that model is a RandomForestClassifier
        self.assertIsInstance(model, RandomForestClassifier)

        # Check that it has learned feature importances
        self.assertGreaterEqual(len(model.feature_importances_), 10)

if __name__ == "__main__":
    unittest.main()