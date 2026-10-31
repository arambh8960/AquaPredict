"""
AquaPredict - Single Sample Prediction Module
Loads Spark MLlib pipelines and generates predictions.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pyspark.ml import PipelineModel
from pyspark.sql.types import DoubleType, StructField, StructType

from src.spark_session import get_spark_session

MODELS_DIR = BASE_DIR / "models"

FEATURE_COLS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

# Model mapping to directory names
MODEL_PATHS = {
    "Random Forest": MODELS_DIR / "random_forest",
    "Decision Tree": MODELS_DIR / "decision_tree",
    "Logistic Regression": MODELS_DIR / "logistic_regression",
    "Naive Bayes": MODELS_DIR / "naive_bayes",
}


class WaterPredictor:

    def __init__(self, spark=None, preload_models: bool = False):
        self.spark = (
            spark if spark else get_spark_session("AquaPredict-Predictor")
        )
        self.models = {}

        if preload_models:
            for name in MODEL_PATHS:
                self._get_model(name)

    def _get_model(self, model_name: str) -> PipelineModel:
        """Loads and caches the model on demand."""
        if model_name not in MODEL_PATHS:
            valid_names = list(MODEL_PATHS.keys())
            raise ValueError(
                f"Invalid model name: '{model_name}'. Available models: {valid_names}"
            )

        if model_name not in self.models:
            model_path = MODEL_PATHS[model_name]
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")
            self.models[model_name] = PipelineModel.load(str(model_path))

        return self.models[model_name]

    def predict(
        self, feature_dict: dict, model_name: str = "Random Forest"
    ) -> dict:
        # 1. Feature validation
        missing_features = [
            col
            for col in FEATURE_COLS
            if col not in feature_dict or feature_dict[col] is None
        ]

        if missing_features:
            raise ValueError(
                f"Missing or null values for features: {missing_features}"
            )

        # 2. Schema aur input DataFrame create karna
        schema = StructType(
            [StructField(col, DoubleType(), False) for col in FEATURE_COLS]
        )

        try:
            input_values = tuple(
                float(feature_dict[col]) for col in FEATURE_COLS
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Features must contain numeric values. Details: {e}"
            )

        input_df = self.spark.createDataFrame([input_values], schema=schema)

        # 3. Model praapt karna (cached/lazy load)
        selected_model = self._get_model(model_name)

        # 4. Prediction execute karna
        transformed_df = selected_model.transform(input_df)
        prediction_res = transformed_df.collect()[0]

        pred_val = int(round(prediction_res["prediction"]))

        # 5. Probability extract karna (safe check)
        prob_potable = None
        if (
            "probability" in prediction_res
            and prediction_res["probability"] is not None
        ):
            try:
                # DenseVector / SparseVector handling
                prob_potable = round(float(prediction_res["probability"][1]), 4)
            except (IndexError, TypeError):
                prob_potable = None

        return {
            "prediction": pred_val,
            "label": (
                "Potable (Safe for Drinking)"
                if pred_val == 1
                else "Non-Potable (Unsafe)"
            ),
            "probability_potable": prob_potable,
            "model_used": model_name,
        }


if __name__ == "__main__":
    predictor = WaterPredictor()

    sample = {
        "ph": 7.2,
        "Hardness": 204.0,
        "Solids": 20791.0,
        "Chloramines": 7.3,
        "Sulfate": 368.0,
        "Conductivity": 564.0,
        "Organic_carbon": 10.3,
        "Trihalomethanes": 86.9,
        "Turbidity": 3.0,
    }

    result = predictor.predict(sample, "Random Forest")
    print("\nTest Prediction Result:", result)