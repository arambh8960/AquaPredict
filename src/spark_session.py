"""
AquaPredict - SparkSession Builder
Path: src/spark_session.py
"""
import sys
from pyspark.sql import SparkSession

def get_spark_session(app_name: str = "AquaPredict-WaterQuality") -> SparkSession:
    """
    Initializes and returns an optimized local SparkSession for macOS.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

if __name__ == "__main__":
    print("Testing SparkSession initialization on macOS...")
    spark = get_spark_session()
    print(f"Spark version: {spark.version}")
    print(f"Master: {spark.sparkContext.master}")
    
    test_df = spark.createDataFrame([(1, "Potable"), (0, "Non-Potable")], ["id", "label"])
    test_df.show()
    spark.stop()
    print("SparkSession successfully verified!")
