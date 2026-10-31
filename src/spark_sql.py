"""
AquaPredict - Spark SQL Module (8 Core Queries)
Path: src/spark_sql.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.spark_session import get_spark_session
from src.preprocessing import load_data, handle_missing_values

def execute_queries(spark):
    raw_df = load_data(spark)
    clean_df = handle_missing_values(raw_df)
    clean_df.createOrReplaceTempView("water_quality")

    queries = [
        ("Query 1: Overall Potability Distribution",
         "SELECT Potability, COUNT(*) AS Total, ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM water_quality),2) AS Pct FROM water_quality GROUP BY Potability"),
        ("Query 2: Parameter Means (Potable vs Non-Potable)",
         "SELECT Potability, ROUND(AVG(ph),2) AS Mean_pH, ROUND(AVG(Hardness),2) AS Mean_Hardness, ROUND(AVG(Solids),2) AS Mean_Solids, ROUND(AVG(Chloramines),2) AS Mean_Chloramines FROM water_quality GROUP BY Potability"),
        ("Query 3: WHO Standard pH Compliance (6.5 - 8.5)",
         "SELECT CASE WHEN ph BETWEEN 6.5 AND 8.5 THEN 'Safe (6.5-8.5)' WHEN ph < 6.5 THEN 'Acidic (<6.5)' ELSE 'Alkaline (>8.5)' END AS Category, COUNT(*) AS Total, SUM(CASE WHEN Potability=1 THEN 1 ELSE 0 END) AS Potable_Count FROM water_quality GROUP BY 1"),
        ("Query 4: High Hardness Water (> 200 mg/L)",
         "SELECT Potability, COUNT(*) AS Count, ROUND(AVG(Hardness),2) AS Avg_Hardness FROM water_quality WHERE Hardness > 200 GROUP BY Potability"),
        ("Query 5: High Turbidity Outliers (> 4.5 NTU)",
         "SELECT ROUND(Turbidity,2) AS Turbidity, ROUND(ph,2) AS pH, Potability FROM water_quality WHERE Turbidity > 4.5 LIMIT 5"),
        ("Query 6: Sulfate and Conductivity Summary",
         "SELECT Potability, ROUND(MIN(Sulfate),2) AS Min_Sulfate, ROUND(MAX(Sulfate),2) AS Max_Sulfate, ROUND(AVG(Conductivity),2) AS Avg_Conductivity FROM water_quality GROUP BY Potability"),
        ("Query 7: Strict WHO Safe Criteria Drinking Water",
         "SELECT COUNT(*) AS Strict_Safe_Count FROM water_quality WHERE ph BETWEEN 6.5 AND 8.5 AND Turbidity < 4.0 AND Chloramines < 8.0 AND Potability = 1"),
        ("Query 8: Organic Carbon & Trihalomethanes Analysis",
         "SELECT Potability, ROUND(AVG(Organic_carbon),2) AS Avg_Carbon, ROUND(AVG(Trihalomethanes),2) AS Avg_Trihalo FROM water_quality GROUP BY Potability")
    ]

    for title, q_str in queries:
        print("\n" + "-" * 60)
        print(title)
        print("-" * 60)
        spark.sql(q_str).show(truncate=False)

if __name__ == "__main__":
    spark = get_spark_session("AquaPredict-SparkSQL")
    execute_queries(spark)
    spark.stop()
