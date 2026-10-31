"""
AquaPredict - Core Spark RDD Operations
Path: src/rdd_operations.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.spark_session import get_spark_session
from src.preprocessing import load_data, handle_missing_values

def run_rdd_demo(spark):
    print("=" * 65)
    print("DEMONSTRATING CORE PYSPARK RDD APIS")
    print("=" * 65)

    sc = spark.sparkContext

    # 1. parallelize() aur collect()
    sample_ph = [6.5, 7.2, 8.1, 5.4, 7.0, 6.8, 9.2, 7.4]
    ph_rdd = sc.parallelize(sample_ph, numSlices=2)
    print("\n1. sc.parallelize() & rdd.collect():")
    print("Sample pH RDD Elements:", ph_rdd.collect())

    # Cleaned DataFrame ko RDD me convert karein
    raw_df = load_data(spark)
    clean_df = handle_missing_values(raw_df)
    water_rdd = clean_df.rdd

    # 2. map()
    potability_pairs_rdd = water_rdd.map(lambda row: (row["Potability"], 1))
    print("\n2. rdd.map() First 5 elements:")
    print(potability_pairs_rdd.take(5))

    # 3. filter()
    potable_rdd = water_rdd.filter(lambda row: row["Potability"] == 1)
    non_potable_rdd = water_rdd.filter(lambda row: row["Potability"] == 0)
    
    p_count = potable_rdd.count()
    np_count = non_potable_rdd.count()
    print(f"\n3. rdd.filter():")
    print(f"Potable Records (Class 1): {p_count}")
    print(f"Non-Potable Records (Class 0): {np_count}")

    # 4. reduce()
    total_p_ph = potable_rdd.map(lambda row: row["ph"]).reduce(lambda a, b: a + b)
    avg_p_ph = total_p_ph / p_count
    
    total_np_ph = non_potable_rdd.map(lambda row: row["ph"]).reduce(lambda a, b: a + b)
    avg_np_ph = total_np_ph / np_count

    print("\n4. rdd.reduce():")
    print(f"Average pH of Potable Water: {avg_p_ph:.3f}")
    print(f"Average pH of Non-Potable Water: {avg_np_ph:.3f}")
    print("=" * 65)

if __name__ == "__main__":
    spark = get_spark_session("AquaPredict-RDD")
    run_rdd_demo(spark)
    spark.stop()
