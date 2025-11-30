import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
BASE_DRIVE = "/content/drive/MyDrive/DEAS_Project"
if not os.path.exists(BASE_DRIVE):
    BASE_DRIVE = "/content/drive/MyDrive/DEAS_project"
INPUT_PATH = os.path.join(BASE_DRIVE, "output", "master_tables", "master_yearly_state.csv")
OUTPUT_PATH = os.path.join(BASE_DRIVE, "output", "dashboard_assets")
os.makedirs(OUTPUT_PATH, exist_ok=True)
print("--- Initializing Transition Engine ---")
spark = SparkSession.builder.appName("TransitionAnalysis").getOrCreate()
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"CRITICAL: Master Yearly Data not found at {INPUT_PATH}")
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
df = df.fillna(0, subset=["total_requirement_mu", "total_renew_gen_mu", "coal_reserves_mt"])
result_schema = StructType([
    StructField("state_name", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("proj_demand_mu", DoubleType(), True),
    StructField("proj_renew_mu", DoubleType(), True),
    StructField("coal_reserves_left_mt", DoubleType(), True),
    StructField("coal_gap_mu", DoubleType(), True)
])
def forecast_transition(pdf):
    state = pdf['state_name'].iloc[0]
    future_years = np.arange(2024, 2041)
    n_years = len(future_years)
    hist_dem = pdf[pdf['total_requirement_mu'] > 0]
    if len(hist_dem) > 1:
        z = np.polyfit(hist_dem['year'], hist_dem['total_requirement_mu'], 1)
        p = np.poly1d(z)
        proj_demand = p(future_years)
    else:
        last_dem = pdf['total_requirement_mu'].max() if len(pdf) > 0 else 0
        proj_demand = np.array([last_dem] * n_years)
    hist_ren = pdf[pdf['total_renew_gen_mu'] > 0]
    if len(hist_ren) > 1:
        start_val = hist_ren['total_renew_gen_mu'].iloc[0]
        end_val = hist_ren['total_renew_gen_mu'].iloc[-1]
        years_diff = hist_ren['year'].iloc[-1] - hist_ren['year'].iloc[0]
        if start_val > 0 and years_diff > 0:
            cagr = (end_val / start_val) ** (1/years_diff) - 1
            cagr = min(max(cagr, 0.05), 0.20)
        else:
            cagr = 0.05
        last_ren = end_val
        proj_renew = [last_ren * ((1 + cagr) ** i) for i in range(1, n_years + 1)]
        proj_renew = np.array(proj_renew)
    else:
        proj_renew = np.array([0.0] * n_years)
    coal_gap = np.maximum(proj_demand - proj_renew, 0)
    current_reserves = pdf['coal_reserves_mt'].max()
    reserves_left = []
    COAL_INTENSITY = 0.0007
    running_reserves = current_reserves
    for gap in coal_gap:
        needed_coal = gap * COAL_INTENSITY
        running_reserves -= needed_coal
        running_reserves = max(running_reserves, 0)
        reserves_left.append(running_reserves)
    return pd.DataFrame({
        'state_name': [state] * n_years,
        'year': future_years,
        'proj_demand_mu': proj_demand,
        'proj_renew_mu': proj_renew,
        'coal_reserves_left_mt': reserves_left,
        'coal_gap_mu': coal_gap
    })
print("Calculating Long-Term Scenarios per State...")
transition_df = df.groupBy("state_name").applyInPandas(forecast_transition, schema=result_schema)
print("Aggregating National View...")
state_out = os.path.join(OUTPUT_PATH, "transition_projections_state.csv")
pdf_state = transition_df.toPandas()
pdf_state.to_csv(state_out, index=False)
print(f"   -> Saved State Projections: {state_out}")
pdf_national = pdf_state.groupby("year")[["proj_demand_mu", "proj_renew_mu", "coal_gap_mu"]].sum().reset_index()
pdf_national['state_name'] = "All India"
national_out = os.path.join(OUTPUT_PATH, "transition_projections_national.csv")
pdf_national.to_csv(national_out, index=False)
print(f"   -> Saved National Projections: {national_out}")
print("\n--- STRATEGIC INSIGHTS ---")
peak_idx = pdf_national['coal_gap_mu'].idxmax()
peak_year = int(pdf_national.loc[peak_idx]['year'])
peak_val = pdf_national.loc[peak_idx]['coal_gap_mu']
print(f" CRITICAL YEAR (Peak Coal): {peak_year}")
print(f"   (Max Coal Demand: {peak_val:,.0f} MU)")
zero_coal = pdf_national[pdf_national['coal_gap_mu'] < 1000]
if not zero_coal.empty:
    target_year = int(zero_coal['year'].min())
    print(f" TARGET YEAR (Net Zero): {target_year}")
else:
    print(" TARGET YEAR: Not achieved by 2040. Faster renewable adoption required.")
print("\n--- Analysis Complete. Ready for Dashboard Tab 5. ---")