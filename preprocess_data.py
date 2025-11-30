import os
from google.colab import drive
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (col, to_date, year, month, sum as _sum,
                                   avg, count, countDistinct, dayofmonth, last_day,
                                   when, coalesce, lit, concat, date_format, mean)
from pyspark.sql.types import StructType
os.system('pip install pyspark')
spark = SparkSession.builder.appName("FinalPipelineV2").config("spark.sql.legacy.timeParserPolicy", "LEGACY").getOrCreate()
drive.mount('/content/drive')
DRIVE_BASE = "/content/drive/MyDrive/DEAS_Project"
INPUT_PATH = os.path.join(DRIVE_BASE, "data")
OUTPUT_PATH = os.path.join(DRIVE_BASE, "output", "processed_individual_v2")
os.makedirs(OUTPUT_PATH, exist_ok=True)
def clean_dates_and_add_meta(df, date_col):
    if df.rdd.isEmpty(): return df
    c = col(date_col)
    df = df.withColumn("parsed_date", coalesce(
        to_date(c, "yyyy-MM-dd"), to_date(c, "dd-MM-yyyy"),
        to_date(c, "M/d/yyyy"), to_date(c, "MM/dd/yyyy"),
        to_date(c, "yyyy-MM-dd HH:mm:ss")
    ))
    df = df.filter(col("parsed_date").isNotNull())
    df = df.withColumn("year", year("parsed_date")) \
           .withColumn("month_num", month("parsed_date")) \
           .withColumn("days_in_month", dayofmonth(last_day(col("parsed_date"))))
    df = df.withColumn("month_start_date",
                       date_format(concat(col("year"), lit("-"), col("month_num"), lit("-01")), "dd-MM-yyyy"))
    return df
def impute_with_monthly_avg(df, target_cols, group_cols=["state_name", "year", "month_num"]):
    w = Window.partitionBy(*group_cols)
    for c in target_cols:
        df = df.withColumn(c, coalesce(col(c), mean(c).over(w), lit(0)))
    return df
def save(df, filename):
    path = os.path.join(OUTPUT_PATH, filename)
    print(f"  -> Saving {filename}...")
    df.write.mode("overwrite").csv(path, header=True)
print("\n[1/10] Processing Coal Stocks (Full Columns & Smart Imputation)...")
df = spark.read.csv(os.path.join(INPUT_PATH, "daily-coal-stocks.csv"), header=True, inferSchema=True)
df = clean_dates_and_add_meta(df, "date")
impute_cols = ["daily_consumption", "daily_requirement", "req_normative_stock", "daily_receipt"]
df = impute_with_monthly_avg(df, impute_cols)
stock_monthly = df.groupBy("state_name", "month_start_date", "year", "month_num", "days_in_month").agg(
    avg("stock_days").alias("avg_coal_stock_days"),
    avg("normative_stock_days").alias("avg_coal_normative_stock_days"),
    avg("req_normative_stock").alias("avg_req_normative_coal_stock_tonnes"),
    avg("total_stock").alias("avg_actual_coal_stock_tonnes"),
    avg("indigenous_stock").alias("avg_indigenous_coal_stock_tonnes"),
    avg("import_stock").alias("avg_import_coal_stock_tonnes"),
    avg("plf_prcnt").alias("avg_coal_plf_prcnt"),
    avg("actual_vs_normative_stock_prcnt").alias("avg_actual_vs_norm_coal_stock_prcnt"),
    _sum("daily_consumption").alias("total_coal_consumption_tonnes"),
    _sum("daily_receipt").alias("total_coal_receipt_tonnes"),
    _sum("daily_requirement").alias("total_coal_requirement_tonnes"),
    count(when(col("is_critical") == "Critical", 1)).alias("coal_stock_critical_incidents"),
    countDistinct("parsed_date").alias("days_reported")
).drop("month_num")
save(stock_monthly, "processed_coal_stocks_monthly")
stock_yearly = stock_monthly.groupBy("state_name", "year").agg(
    avg("avg_coal_stock_days").alias("avg_coal_stock_days"),
    avg("avg_coal_normative_stock_days").alias("avg_normative_coal_stock_days"),
    avg("avg_req_normative_coal_stock_tonnes").alias("avg_req_normative_coal_stock_tonnes"),
    avg("avg_import_coal_stock_tonnes").alias("avg_import_coal_stock_tonnes"),
    avg("avg_indigenous_coal_stock_tonnes").alias("avg_indigenous_coal_stock_tonnes"),
    avg("avg_coal_plf_prcnt").alias("avg_coal_plf_prcnt"),
    avg("avg_actual_vs_norm_coal_stock_prcnt").alias("avg_actual_vs_norm_coal_stock_prcnt"),
    _sum("total_coal_consumption_tonnes").alias("total_coal_consumption_tonnes"),
    _sum("total_coal_receipt_tonnes").alias("total_coal_receipt_tonnes"),
    _sum("total_coal_requirement_tonnes").alias("total_coal_requirement_tonnes"),
    avg("avg_actual_coal_stock_tonnes").alias("avg_actual_coal_stock_tonnes"),
    _sum("coal_stock_critical_incidents").alias("total_coal_stock_critical_incidents"),
    countDistinct("month_start_date").alias("months_reported")
)
save(stock_yearly, "processed_coal_stocks_yearly")
print("\n[2/10] Processing Installed Capacity (Full Columns)...")
df = spark.read.csv(os.path.join(INPUT_PATH, "installed-capacity-statewise.csv"), header=True, inferSchema=True)
df = clean_dates_and_add_meta(df, "date").fillna(0)
cap_monthly = df.groupBy("state_name", "month_start_date", "year", "month_num", "days_in_month").agg(
    _sum("coal_cap").alias("cap_coal_mw"),
    _sum("gas_cap").alias("cap_gas_mw"),
    _sum("diesel_cap").alias("cap_diesel_mw"),
    _sum("lignite_cap").alias("cap_lignite_mw"),
    _sum("nuclear_cap").alias("cap_nuclear_mw"),
    _sum("hydro_cap").alias("cap_hydro_mw"),
    _sum("res_cap").alias("cap_res_mw")
).withColumn("total_thermal_mw", col("cap_coal_mw") + col("cap_gas_mw") + col("cap_diesel_mw") + col("cap_lignite_mw")) \
 .withColumn("total_capacity_mw", col("total_thermal_mw") + col("cap_nuclear_mw") + col("cap_hydro_mw") + col("cap_res_mw")) \
 .drop("month_num")
save(cap_monthly, "processed_capacity_monthly")
cap_yearly = cap_monthly.groupBy("state_name", "year").agg(
    avg("total_capacity_mw").alias("avg_total_capacity_mw"),
    avg("total_thermal_mw").alias("avg_thermal_capacity_mw"),
    avg("cap_nuclear_mw").alias("avg_nuclear_capacity__mw"),
    avg("cap_hydro_mw").alias("avg_hydro_capacity__mw"),
    avg("cap_res_mw").alias("avg_res_capacity_mw"),
    countDistinct("month_start_date").alias("months_reported")
)
save(cap_yearly, "processed_capacity_yearly")
print("\n[3/10] Processing Power Generation (Full Columns)...")
df = spark.read.csv(os.path.join(INPUT_PATH, "daily-power-generation.csv"), header=True, inferSchema=True)
df = clean_dates_and_add_meta(df, "date").fillna(0, subset=["todays_gen_act", "todays_gen_prgm", "monitored_capacity"])
gen_monthly = df.groupBy("state_name", "month_start_date", "year", "month_num", "days_in_month").agg(
    _sum("todays_gen_act").alias("total_gen_actual_mu"),
    _sum("todays_gen_prgm").alias("total_gen_program_mu"),
    avg("monitored_capacity").alias("avg_monitored_cap_mw"),
    countDistinct("parsed_date").alias("days_reported")
).drop("month_num")
save(gen_monthly, "processed_generation_monthly")
gen_yearly = gen_monthly.groupBy("state_name", "year").agg(
    _sum("total_gen_actual_mu").alias("total_gen_mu"),
    avg("avg_monitored_cap_mw").alias("avg_capacity_mw"),
    countDistinct("month_start_date").alias("months_reported")
)
save(gen_yearly, "processed_generation_yearly")
print("\n[4/10] Processing Renewable Generation (Full Columns)...")
df = spark.read.csv(os.path.join(INPUT_PATH, "daily-renewable-energy-generation.csv"), header=True, inferSchema=True)
df = clean_dates_and_add_meta(df, "date").fillna(0)
ren_monthly = df.groupBy("state_name", "month_start_date", "year", "month_num", "days_in_month").agg(
    _sum("total_renewable_energy").alias("total_renew_gen_mu"),
    _sum("solar_energy").alias("total_solar_gen_mu"),
    _sum("wind_energy").alias("total_wind_gen_mu"),
    _sum("other_renewable_energy").alias("total_other_gen_mu")
).drop("month_num")
save(ren_monthly, "processed_renewable_monthly")
ren_yearly = ren_monthly.groupBy("state_name", "year").agg(
    _sum("total_renew_gen_mu").alias("total_renew_gen_mu"),
    _sum("total_solar_gen_mu").alias("total_solar_gen_mu"),
    _sum("total_wind_gen_mu").alias("total_wind_gen_mu"),
    _sum("total_other_gen_mu").alias("total_other_gen_mu"),
    countDistinct("month_start_date").alias("months_reported")
)
save(ren_yearly, "processed_renewable_yearly")
def clean_dates_and_add_meta_outage(df):
    if df.rdd.isEmpty(): return df
    def parse_date_col(c):
        return coalesce(
            to_date(c, "dd-MM-yyyy"),
            to_date(c, "yyyy-MM-dd"),
            to_date(c, "M/d/yyyy"),
            to_date(c, "MM/dd/yyyy")
        )
    df = df.withColumn("report_date", parse_date_col(col("date"))) \
           .withColumn("start_date", parse_date_col(col("outage_date")))
    df = df.filter(col("report_date").isNotNull())
    df = df.withColumn("year", year("report_date")) \
           .withColumn("month_num", month("report_date")) \
           .withColumn("days_in_month", dayofmonth(last_day(col("report_date")))) \
           .withColumn("month_start_date", date_format(concat(col("year"), lit("-"), col("month_num"), lit("-01")), "dd-MM-yyyy"))
    df = df.withColumn("unique_incident_id",
                       concat(col("power_station"), lit("_"), col("power_station_unit"), lit("_"), coalesce(col("start_date"), lit("Unknown"))))
    df = df.withColumn("is_carry_over",
                       when(col("start_date") < col("report_date"), 1).otherwise(0))
    return df
print("\n[5/10] Processing Power Outages...")
df_raw = spark.read.csv(os.path.join(INPUT_PATH, "daily-power-outage.csv"), header=True, inferSchema=True)
df_out = clean_dates_and_add_meta_outage(df_raw)
daily_outage = df_out.groupBy("state_name", "month_start_date", "year", "month_num", "days_in_month", "report_date").agg(
    _sum("cap_under_outage").alias("total_daily_mw_down"),
    _sum("monitored_capacity").alias("total_daily_monitored_cap"),
    _sum(when(col("is_carry_over") == 0, col("cap_under_outage")).otherwise(0)).alias("fresh_daily_mw_down")
)
monthly_outage_stats = daily_outage.groupBy("state_name", "month_start_date", "year", "month_num", "days_in_month").agg(
    avg("total_daily_mw_down").alias("avg_total_outage_mw"),
    avg("fresh_daily_mw_down").alias("avg_fresh_outage_mw"),
    avg("total_daily_monitored_cap").alias("avg_monitored_cap_mw"),
    _sum("total_daily_mw_down").alias("sum_mw_days")
).withColumn("total_energy_lost_mwh", col("sum_mw_days") * 24).drop("sum_mw_days")
incident_counts = df_out.groupBy("state_name", "month_start_date", "year").agg(
    countDistinct("unique_incident_id").alias("total_incidents_count"),
    countDistinct(when(col("is_carry_over") == 0, col("unique_incident_id"))).alias("fresh_incidents_count")
)
out_monthly = monthly_outage_stats.join(incident_counts, on=["state_name", "month_start_date", "year"], how="left").drop("month_num")
print(f"Saving Monthly Outage Data...")
out_monthly.write.mode("overwrite").csv(os.path.join(OUTPUT_PATH, "processed_outage_monthly"), header=True)
out_yearly = out_monthly.groupBy("state_name", "year").agg(
    avg("avg_total_outage_mw").alias("yearly_avg_total_outage_mw"),
    avg("avg_fresh_outage_mw").alias("yearly_avg_fresh_outage_mw"),
    _sum("total_energy_lost_mwh").alias("total_energy_lost_mwh"),
    _sum("fresh_incidents_count").alias("total_fresh_incidents"),
    countDistinct("month_start_date").alias("months_reported")
)
print(f"Saving Yearly Outage Data...")
out_yearly.write.mode("overwrite").csv(os.path.join(OUTPUT_PATH, "processed_outage_yearly"), header=True)
print("\n[6/10] Processing Requirements (Full Columns)...")
df = spark.read.csv(os.path.join(INPUT_PATH, "energy-requirement-and-availabililty.csv"), header=True, inferSchema=True)
df = clean_dates_and_add_meta(df, "month")
req_monthly = df.withColumn("deficit_mu", col("energy_requirement") - col("energy_availability")) \
    .withColumn("deficit_pct", (col("deficit_mu") / col("energy_requirement")) * 100) \
    .select("state_name", "month_start_date", "year", "days_in_month", "energy_requirement", "energy_availability", "deficit_mu", "deficit_pct")
save(req_monthly, "processed_requirements_monthly")
req_yearly = req_monthly.groupBy("state_name", "year").agg(
    _sum("energy_requirement").alias("total_requirement_mu"),
    _sum("energy_availability").alias("total_availability_mu"),
    _sum("deficit_mu").alias("total_deficit_mu"),
    countDistinct("month_start_date").alias("months_reported")
)
save(req_yearly, "processed_requirements_yearly")
print("\nProcessing Yearly Static Files...")
df = spark.read.csv(os.path.join(INPUT_PATH, "coal-reserves.csv"), header=True, inferSchema=True)
df.groupBy("state_name", "year").agg(_sum("quantity").alias("coal_reserves_mt")) \
  .write.mode("overwrite").csv(os.path.join(OUTPUT_PATH, "processed_coal_reserves_yearly"), header=True)
df = spark.read.csv(os.path.join(INPUT_PATH, "domestic-coal-production.csv"), header=True, inferSchema=True)
df.groupBy("state_name", "year").agg(_sum("quantity").alias("coal_production_mt")) \
  .write.mode("overwrite").csv(os.path.join(OUTPUT_PATH, "processed_coal_production_yearly"), header=True)
df = spark.read.csv(os.path.join(INPUT_PATH, "coal-consumption.csv"), header=True, inferSchema=True)
df.filter(col("sector").contains("Power") | col("sector").contains("Utility")) \
  .groupBy("state_name", "year").agg(_sum("quantity").alias("coal_consumption_power_mt")) \
  .write.mode("overwrite").csv(os.path.join(OUTPUT_PATH, "processed_coal_consumption_yearly"), header=True)
df = spark.read.csv(os.path.join(INPUT_PATH, "coal-import.csv"), header=True, inferSchema=True)
df.groupBy("year").agg(_sum("quantity").alias("total_coal_import_mt"), avg("price").alias("avg_import_price(Rs.)"), _sum("amount").alias("total_amount(Cr)")) \
  .write.mode("overwrite").csv(os.path.join(OUTPUT_PATH, "processed_coal_imports_yearly"), header=True)
