import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, StringType
from pyspark.sql.functions import (
    col, from_json, to_timestamp, trim, lower, when
)
import os

OUT_BASE = r"D:\data"

ALL_PATH = os.path.join(OUT_BASE, "stream_all_by_state")
ALERTS_PATH = os.path.join(OUT_BASE, "alerts")

CKPT_ALL = os.path.join(OUT_BASE, "checkpoints_all")
CKPT_ALERTS = os.path.join(OUT_BASE, "checkpoints_alerts")

schema = StructType([
    StructField("id", StringType()),
    StructField("date", StringType()),
    StructField("state_name", StringType()),
    StructField("state_code", StringType()),
    StructField("power_station_name", StringType()),
    StructField("sector", StringType()),
    StructField("utility", StringType()),
    StructField("mode_of_transport", StringType()),
    StructField("capacity", StringType()),
    StructField("daily_requirement", StringType()),
    StructField("daily_receipt", StringType()),
    StructField("daily_consumption", StringType()),
    StructField("req_normative_stock", StringType()),
    StructField("normative_stock_days", StringType()),
    StructField("indigenous_stock", StringType()),
    StructField("import_stock", StringType()),
    StructField("total_stock", StringType()),
    StructField("stock_days", StringType()),
    StructField("plf_prcnt", StringType()),
    StructField("actual_vs_normative_stock_prcnt", StringType()),
    StructField("is_critical", StringType()),
    StructField("remarks", StringType()),
])

spark = (SparkSession.builder
    .appName("CoalStockReceiverFullSchema")
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1")
    .config("spark.sql.shuffle.partitions", "6")
    .config("spark.sql.parquet.mergeSchema", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

raw_kafka = (spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "127.0.0.1:9092")
    .option("subscribe", "daily_coal_events")
    .option("startingOffsets", "latest")
    .load()
)

df_raw = raw_kafka.selectExpr(
    "CAST(value AS STRING) AS raw_value",
    "timestamp AS kafka_ingest_ts"
)

df = df_raw.withColumn("json", from_json(col("raw_value"), schema)) \
           .select("json.*", "kafka_ingest_ts")

for f in schema.fields:
    if isinstance(f.dataType, StringType):
        df = df.withColumn(f.name, trim(col(f.name)))

df = df.withColumn("event_ts", to_timestamp(col("date")))

df = df.withColumn(
    "severity",
    when(col("is_critical").isNull(), "NONE")
    .when(lower(col("is_critical")).rlike("super"), "SUPER")
    .when(lower(col("is_critical")).rlike("critical"), "CRITICAL")
    .otherwise("NONE")
)

stream_writer = (df.writeStream
    .format("parquet")
    .option("path", ALL_PATH)
    .option("checkpointLocation", CKPT_ALL)
    .partitionBy("state_name")
    .outputMode("append")
    .trigger(processingTime="10 seconds")
    .start()
)

alerts = df.filter(col("severity") != "NONE")

alerts_writer = (alerts.writeStream
    .format("parquet")
    .option("path", ALERTS_PATH)
    .option("checkpointLocation", CKPT_ALERTS)
    .outputMode("append")
    .trigger(processingTime="10 seconds")
    .start()
)

print("Stream receiving...")
try:
    spark.streams.awaitAnyTermination()
except KeyboardInterrupt:
    print("Stopping streams...")
    for q in spark.streams.active:
        q.stop()
    spark.stop()
    sys.exit(0)

