"""
AquaPredict - PySpark Data Preprocessing & Cleaning
Path: src/preprocessing.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan, count
from pyspark.ml.feature import Imputer
from src.spark_session import get_spark_session

DATA_PATH = BASE_DIR / "data" / "water_potability.csv"

def load_data(spark) -> DataFrame:
    """Loads Kaggle CSV into PySpark DataFrame."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Check data/ folder.")
    
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(DATA_PATH))
    )
    return df

def check_missing_values(df: DataFrame):
    """Prints count of missing / null values per column."""
    print("\n--- Missing Values Count Per Column ---")
    missing_exprs = [
        count(when(col(c).isNull() | isnan(col(c)), c)).alias(c)
        for c in df.columns
    ]
    df.select(missing_exprs).show()

def handle_missing_values(df: DataFrame) -> DataFrame:
    """
    Imputes missing values using Spark MLlib Imputer (Mean strategy).
    Target column 'Potability' is preserved as is.
    """
    feature_cols = [c for c in df.columns if c != "Potability"]
    imputed_cols = [f"{c}_imputed" for c in feature_cols]
    
    imputer = Imputer(
        inputCols=feature_cols,
        outputCols=imputed_cols,
        strategy="mean"
    )
    
    model = imputer.fit(df)
    imputed_df = model.transform(df)
    
    cleaned_df = imputed_df.drop(*feature_cols)
    for c in feature_cols:
        cleaned_df = cleaned_df.withColumnRenamed(f"{c}_imputed", c)
        
    return cleaned_df.select(feature_cols + ["Potability"])

if __name__ == "__main__":
    spark = get_spark_session("AquaPredict-Preprocessing")
    print("1. Loading raw dataset into PySpark...")
    raw_df = load_data(spark)
    
    print("\n2. Dataset Schema:")
    raw_df.printSchema()
    
    total_count = raw_df.count()
    print(f"\nTotal Records: {total_count}")
    
    print("\n3. Missing Values before cleaning:")
    check_missing_values(raw_df)
    
    print("4. Applying Spark MLlib Imputer for missing value imputation...")
    cleaned_df = handle_missing_values(raw_df)
    
    print("\n5. Missing Values after cleaning (all must be 0):")
    check_missing_values(cleaned_df)
    
    print("6. Sample Cleaned Records (First 3 rows):")
    cleaned_df.show(3, truncate=False)
    
    spark.stop()
    print("\nPhase 2 Preprocessing & Data Cleaning completed successfully!")
