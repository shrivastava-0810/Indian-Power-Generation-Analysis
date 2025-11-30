import os
import json
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
INPUT_PATH = "/content/drive/MyDrive/DEAS_Project/output/master_tables/master_monthly_state.csv"
OUTPUT_PATH = "/content/drive/MyDrive/DEAS_Project/output/dashboard_assets"
os.makedirs(OUTPUT_PATH, exist_ok=True)
print("--- Initializing Master AI Engine ---")
spark = SparkSession.builder.appName("MasterAIEngine").getOrCreate()
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
if "join_date" in df.columns:
    df = df.withColumn("date", F.col("join_date").cast("date"))
elif "month_start_date" in df.columns:
    df = df.withColumn("date", F.col("month_start_date").cast("date"))
df = df.withColumn("year", F.year("date")).withColumn("month", F.month("date"))
print("Building Global Features (Lags, Rolling, Seasonality)...")
w_state = Window.partitionBy("state_name").orderBy("date")
w_rolling_3 = Window.partitionBy("state_name").orderBy("date").rowsBetween(-2, 0)
w_rolling_12 = Window.partitionBy("state_name").orderBy("date").rowsBetween(-11, 0)
df = df.withColumn("month_sin", F.sin(2 * np.pi * F.col("month") / 12)) \
       .withColumn("month_cos", F.cos(2 * np.pi * F.col("month") / 12))
df = df.withColumn("demand_lag_1", F.lag("energy_requirement", 1).over(w_state)) \
       .withColumn("demand_lag_12", F.lag("energy_requirement", 12).over(w_state))
gen_col = "total_gen_actual_mu"
df = df.withColumn("gen_lag_1", F.lag(gen_col, 1).over(w_state)) \
       .withColumn("gen_lag_12", F.lag(gen_col, 12).over(w_state)) \
       .withColumn("gen_roll_3", F.mean(gen_col).over(w_rolling_3)) \
       .withColumn("gen_roll_12", F.mean(gen_col).over(w_rolling_12))
df = df.withColumn("target_demand_next", F.lead("energy_requirement", 1).over(w_state)) \
       .withColumn("target_gen_next", F.lead(gen_col, 1).over(w_state))
df = df.fillna(0, subset=["total_renew_gen_mu", "avg_total_outage_mw", "gen_lag_1"])
df = df.fillna(0, subset=["total_capacity_mw", "avg_coal_stock_days"])
df.cache()
print(f"Data Prepared. Total Rows: {df.count()}")
print("\n--- Training Model A: Deficit Risk (Snapshot) ---")
features_A = [
    "total_capacity_mw", "avg_coal_stock_days", "avg_total_outage_mw",
    "total_renew_gen_mu", "month_sin", "month_cos"
]
df_A = df.dropna(subset=features_A + ["deficit_pct"])
assembler_A = VectorAssembler(inputCols=features_A, outputCol="features")
rf = RandomForestRegressor(featuresCol="features", labelCol="deficit_pct", numTrees=100, seed=42)
pipeline_A = Pipeline(stages=[assembler_A, rf])
model_A = pipeline_A.fit(df_A)
print(" Model A Trained.")
model_A.write().overwrite().save(os.path.join(OUTPUT_PATH, "model_a_deficit"))
with open(os.path.join(OUTPUT_PATH, "features_a.json"), "w") as f: json.dump(features_A, f)
print("\n--- Training Model B: Demand Forecast (Time-Series) ---")
features_B = ["demand_lag_1", "demand_lag_12", "month_sin", "month_cos"]
df_B = df.dropna(subset=features_B + ["target_demand_next"])
train_B = df_B.filter(F.col("year") < 2023)
test_B = df_B.filter(F.col("year") >= 2023)
assembler_B = VectorAssembler(inputCols=features_B, outputCol="features")
gbt_B = GBTRegressor(featuresCol="features", labelCol="target_demand_next", maxIter=50, seed=42)
pipeline_B = Pipeline(stages=[assembler_B, gbt_B])
model_B = pipeline_B.fit(train_B)
preds_B = model_B.transform(test_B)
rmse_B = RegressionEvaluator(labelCol="target_demand_next", metricName="rmse").evaluate(preds_B)
print(f" Model B Trained. RMSE: {rmse_B:.2f} MU")
model_B.write().overwrite().save(os.path.join(OUTPUT_PATH, "model_b_demand"))
print("\n--- Training Model C: Generation Forecast (Advanced) ---")
indexer = StringIndexer(inputCol="state_name", outputCol="state_idx", handleInvalid="keep")
encoder = OneHotEncoder(inputCols=["state_idx"], outputCols=["state_vec"])
features_C = [
    "gen_lag_1", "gen_lag_12",
    "gen_roll_3", "gen_roll_12",
    "month_sin", "month_cos",
    "state_vec",
    "avg_coal_stock_days"
]
df_C = df.dropna(subset=["gen_lag_12", "gen_roll_12", "target_gen_next", "avg_coal_stock_days"])
train_C = df_C.filter(F.col("year") < 2023)
test_C = df_C.filter(F.col("year") >= 2023)
assembler_C = VectorAssembler(inputCols=features_C, outputCol="features")
gbt_C = GBTRegressor(featuresCol="features", labelCol="target_gen_next", maxIter=50, seed=42)
pipeline_C = Pipeline(stages=[indexer, encoder, assembler_C, gbt_C])
model_C = pipeline_C.fit(train_C)
preds_C = model_C.transform(test_C)
rmse_C = RegressionEvaluator(labelCol="target_gen_next", metricName="rmse").evaluate(preds_C)
print(f" Model C Trained. RMSE: {rmse_C:.2f} MU")
model_C.write().overwrite().save(os.path.join(OUTPUT_PATH, "model_c_generation"))
print("\nExporting Sample Forecasts...")
sample_df = preds_C.select("state_name", "date", "target_gen_next", "prediction") \
                   .withColumnRenamed("target_gen_next", "Actual_Gen") \
                   .withColumnRenamed("prediction", "Forecast_Gen")
preds_B_clean = preds_B.select("state_name", "date", "target_demand_next", "prediction") \
                       .withColumnRenamed("target_demand_next", "Actual_Demand") \
                       .withColumnRenamed("prediction", "Forecast_Demand")
final_samples = sample_df.join(preds_B_clean, on=["state_name", "date"], how="inner")
final_samples.toPandas().to_csv(os.path.join(OUTPUT_PATH, "forecast_samples.csv"), index=False)
print("--- All Assets Saved. Ready for Dashboard! ---")