import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
BASE_DRIVE = "/content/drive/MyDrive/DEAS_Project"
if not os.path.exists(BASE_DRIVE):
    BASE_DRIVE = "/content/drive/MyDrive/DEAS_project"
PATH_MONTHLY = os.path.join(BASE_DRIVE, "output", "master_tables", "master_monthly_state.csv")
OUTPUT_PATH  = os.path.join(BASE_DRIVE, "output", "dashboard_assets")
MODELS_PATH  = os.path.join(OUTPUT_PATH, "models")
FORECAST_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "forecast_results")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(FORECAST_OUTPUT_PATH, exist_ok=True)
FORECAST_TILL_YEAR = 2050
TARGET_COLUMNS = [
    "avg_total_outage_mw",
    "avg_fresh_outage_mw",
    "total_incidents_count",
    "fresh_incidents_count",
    "avg_actual_coal_stock_tonnes",
    "avg_indigenous_coal_stock_tonnes",
    "avg_import_coal_stock_tonnes",
    "coal_stock_critical_incidents",
    "energy_requirement",
    "energy_availability",
    "total_gen_actual_mu",
    "avg_coal_stock_days",
    "avg_coal_normative_stock_days",
    "total_amountCr"
]
STATE_COL_NAME = "state_name"
YEAR_COL = "year"
MONTH_COL = "month_num"
print("--- Initializing Strategic Engine V5 (Memory Optimized) ---")
spark = SparkSession.builder \
    .appName("TransitionAnalysisV5_Optimized") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
print(f"Loading data from {PATH_MONTHLY}...")
df_monthly = spark.read.csv(PATH_MONTHLY, header=True, inferSchema=True)
df_prep = df_monthly.withColumn(YEAR_COL, F.col(YEAR_COL).cast(IntegerType())) \
                    .withColumn(MONTH_COL, F.col(MONTH_COL).cast(IntegerType()))
valid_targets = []
for col in TARGET_COLUMNS:
    if col in df_prep.columns:
        df_prep = df_prep.withColumn(col, F.col(col).cast(DoubleType()))
        valid_targets.append(col)
    else:
        print(f"WARNING: Column {col} not found. Skipped.")
print("Calculating 'All India' aggregates...")
df_states_only = df_prep.filter(
    (~F.lower(F.col(STATE_COL_NAME)).isin(["all india", "total", "india"]))
)
agg_exprs = []
for col in valid_targets:
    if "days" in col.lower() or "normative" in col.lower() or "pct" in col.lower():
        agg_exprs.append(F.avg(col).alias(col))
    else:
        agg_exprs.append(F.sum(col).alias(col))
df_all_india = df_states_only.groupBy(YEAR_COL, MONTH_COL).agg(*agg_exprs) \
                             .withColumn(STATE_COL_NAME, F.lit("All India"))
common_cols = [STATE_COL_NAME, YEAR_COL, MONTH_COL] + valid_targets
df_full_history = df_states_only.select(common_cols).unionByName(df_all_india.select(common_cols))
max_history_year = df_full_history.agg({YEAR_COL: "max"}).collect()[0][0]
min_history_year = df_full_history.agg({YEAR_COL: "min"}).collect()[0][0]
print(f"History Range: {min_history_year} - {max_history_year}")
indexer = StringIndexer(inputCol=STATE_COL_NAME, outputCol="State_Index", handleInvalid="keep")
encoder = OneHotEncoder(inputCols=["State_Index"], outputCols=["State_Vec"])
assembler = VectorAssembler(inputCols=[YEAR_COL, MONTH_COL, "State_Vec"], outputCol="features")
pipeline_prep = Pipeline(stages=[indexer, encoder])
model_prep = pipeline_prep.fit(df_full_history)
df_features = model_prep.transform(df_full_history)
print("Generating Master Time Grid...")
unique_states = [row[STATE_COL_NAME] for row in df_full_history.select(STATE_COL_NAME).distinct().collect()]
grid_data = []
for year in range(min_history_year, FORECAST_TILL_YEAR + 1):
    for month in range(1, 13):
        for state in unique_states:
            grid_data.append({STATE_COL_NAME: state, YEAR_COL: year, MONTH_COL: month})
df_grid = spark.createDataFrame(pd.DataFrame(grid_data))
df_grid = df_grid.withColumn(YEAR_COL, F.col(YEAR_COL).cast(IntegerType())) \
                 .withColumn(MONTH_COL, F.col(MONTH_COL).cast(IntegerType()))
df_grid_features = model_prep.transform(df_grid)
print("Initializing Pandas Master DataFrame...")
pdf_master = df_grid.select(STATE_COL_NAME, YEAR_COL, MONTH_COL).toPandas()
pdf_master.sort_values(by=[STATE_COL_NAME, YEAR_COL, MONTH_COL], inplace=True)
for target in valid_targets:
    print(f"--- Processing: {target} ---")
    df_valid_history = df_features.filter((F.col(target).isNotNull()) & (F.col(target) > 0))
    if df_valid_history.count() < 24:
        print(f"  Skipping {target}: Insufficient data.")
        pdf_master[target] = 0.0
        continue
    lr = LinearRegression(
        featuresCol="features",
        labelCol=target,
        predictionCol=f"pred_{target}",
        standardization=True
    )
    model = lr.fit(assembler.transform(df_valid_history))
    model.write().overwrite().save(os.path.join(MODELS_PATH, f"lr_model_{target}"))
    predictions = model.transform(assembler.transform(df_grid_features))
    df_history_subset = df_valid_history.select(STATE_COL_NAME, YEAR_COL, MONTH_COL, F.col(target).alias("actual_val"))
    df_pred_subset = predictions.select(STATE_COL_NAME, YEAR_COL, MONTH_COL, F.col(f"pred_{target}").alias("pred_val"))
    df_merged = df_grid.select(STATE_COL_NAME, YEAR_COL, MONTH_COL) \
        .join(df_pred_subset, on=[STATE_COL_NAME, YEAR_COL, MONTH_COL], how="left") \
        .join(df_history_subset, on=[STATE_COL_NAME, YEAR_COL, MONTH_COL], how="left")
    df_merged = df_merged.withColumn(
        target,
        F.greatest(F.coalesce(F.col("actual_val"), F.col("pred_val")), F.lit(0.0))
    )
    pdf_col = df_merged.select(STATE_COL_NAME, YEAR_COL, MONTH_COL, target).toPandas()
    pdf_master = pd.merge(pdf_master, pdf_col, on=[STATE_COL_NAME, YEAR_COL, MONTH_COL], how="left")
    pdf_master[target] = pdf_master[target].fillna(0.0)
print("Applying logical constraints in Pandas...")
if "avg_total_outage_mw" in pdf_master.columns and "avg_fresh_outage_mw" in pdf_master.columns:
    pdf_master["avg_fresh_outage_mw"] = np.minimum(pdf_master["avg_fresh_outage_mw"], pdf_master["avg_total_outage_mw"])
if "total_incidents_count" in pdf_master.columns and "fresh_incidents_count" in pdf_master.columns:
    pdf_master["fresh_incidents_count"] = np.minimum(pdf_master["fresh_incidents_count"], pdf_master["total_incidents_count"])
int_cols = [
    "total_incidents_count", "fresh_incidents_count", "coal_stock_critical_incidents",
    "avg_coal_stock_days", "avg_coal_normative_stock_days"
]
for c in int_cols:
    if c in pdf_master.columns:
        pdf_master[c] = pdf_master[c].round().astype(int)
pdf_master['date_ref'] = pd.to_datetime(pdf_master[YEAR_COL].astype(str) + '-' + pdf_master[MONTH_COL].astype(str) + '-01')
pdf_master['Type'] = np.where(pdf_master[YEAR_COL] <= max_history_year, 'Historical', 'Forecast')
pdf_master = pdf_master.sort_values(by=[STATE_COL_NAME, YEAR_COL, MONTH_COL])
save_path = os.path.join(FORECAST_OUTPUT_PATH, "master_forecast_combined.csv")
print(f"Saving to: {save_path}")
pdf_master.to_csv(save_path, index=False)
print("SUCCESS.")