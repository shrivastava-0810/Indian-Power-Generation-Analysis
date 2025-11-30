import csv, json, time, os
from kafka import KafkaProducer

BASE_DIR = r"D:\IISc_Mtech\projects\DEAS\data"
CSV_PATH = os.path.join(BASE_DIR, "daily-coal-stocks.csv")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "daily_coal_events")
THROTTLE_SEC = float(os.environ.get("THROTTLE_SEC", "0.1"))

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=5
)

def row_to_message(row: dict):
    if "coal_stock_tonnes" in row:
        try:
            row["coal_stock_tonnes"] = float(row["coal_stock_tonnes"]) if row["coal_stock_tonnes"] != "" else None
        except:
            row["coal_stock_tonnes"] = None
    return row

with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        msg = row_to_message(row)
        producer.send(TOPIC, msg)
        producer.flush()
        print("sent:", msg.get("power_station_name"), msg.get("state_name"), msg.get("date"))
        time.sleep(THROTTLE_SEC)

print("done")
