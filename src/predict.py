"""
AquaPredict - Single Sample Prediction Module
Loads saved Spark MLlib models and generates predictions.
Path: src/predict.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScalerModel
from pyspark.ml.classification import (
    LogisticRegressionModel,
    DecisionTreeClassificationModel,
    NaiveBayesModel,
    RandomForestClassificationModel
)
from src.spark_session import get_spark_session

MODELS_DIR = BASE_DIR / "models"
FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines",
    "Sulfate", "Conductivity", "Organic_carbon",
    "Trihalomethanes", "Turbidity"
]

class WaterPredictor:
    def __init__(self, spark=None):
        self.spark = spark if spark else get_spark_session("AquaPredict-Predictor")
        self.scaler = MinMaxScalerModel.load(str(MODELS_DIR / "scaler_model"))
        self.assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
        
        self.models = {
            "Random Forest": RandomForestClassificationModel.load(str(MODELS_DIR / "random_forest")),
            "Decision Tree": DecisionTreeClassificationModel.load(str(MODELS_DIR / "decision_tree")),
            "Logistic Regression": LogisticRegressionModel.load(str(MODELS_DIR / "logistic_regression")),
            "Naive Bayes": NaiveBayesModel.load(str(MODELS_DIR / "naive_bayes"))
        }

    def predict(self, feature_dict: dict, model_name: str = "Random Forest"):
        schema = StructType([StructField(c, DoubleType(), True) for c in FEATURE_COLS])
        input_df = self.spark.createDataFrame([tuple(float(feature_dict[c]) for c in FEATURE_COLS)], schema=schema)

        assembled_df = self.assembler.transform(input_df)
        scaled_df = self.scaler.transform(assembled_df)

        selected_model = self.models.get(model_name, self.models["Random Forest"])
        prediction_res = selected_model.transform(scaled_df).collect()[0]

        pred_val = int(prediction_res["prediction"])
        prob_potable = None
        if "probability" in prediction_res:
            prob_potable = float(prediction_res["probability"][1])

        return {
            "prediction": pred_val,
            "label": "Potable (Safe for Drinking)" if pred_val == 1 else "Non-Potable (Unsafe)",
            "probability_potable": prob_potable,
            "model_used": model_name
        }

if __name__ == "__main__":
    predictor = WaterPredictor()
    sample = {
        "ph": 7.2, "Hardness": 204.0, "Solids": 20791.0,
        "Chloramines": 7.3, "Sulfate": 368.0, "Conductivity": 564.0,
        "Organic_carbon": 10.3, "Trihalomethanes": 86.9, "Turbidity": 3.0
    }
    res = predictor.predict(sample, "Random Forest")
    print("\nTest Prediction Result:", res)
