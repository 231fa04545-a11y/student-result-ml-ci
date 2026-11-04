
import os
import json
import unittest
import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("student_result_model.pkl")

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_model_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_range(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_prediction_output(self):
        sample = pd.DataFrame([{
            "attendance": 85,
            "internal_marks": 75,
            "assignment_marks": 80,
            "previous_score": 78
        }])

        prediction = self.model.predict(sample)[0]
        self.assertIn(prediction, [0, 1])

    def test_high_performance_student(self):
        sample = pd.DataFrame([{
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80
        }])

        prediction = self.model.predict(sample)[0]
        self.assertEqual(prediction, 1)

    def test_low_performance_student(self):
        sample = pd.DataFrame([{
            "attendance": 55,
            "internal_marks": 30,
            "assignment_marks": 40,
            "previous_score": 35
        }])

        prediction = self.model.predict(sample)[0]
        self.assertEqual(prediction, 0)


if __name__ == "__main__":
    unittest.main()
