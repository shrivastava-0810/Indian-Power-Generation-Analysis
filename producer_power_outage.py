# producer.py
import json
import time
import pandas as pd
from datetime import datetime, timezone
from kafka import KafkaProducer
import os

# -------- CONFIG ----------
KAFKA_BROKER = "localhost:9092"
TOPIC = "telemetry"

CSV_PATH = r"F:\DataEngineering\Project\LiveProject\processed_daily_power_outage.csv"
SLEEP_SECONDS = 3
REPLAY = False
WRITE_FALLBACK = True

# --------- helper ----------
def to_iso(dt):
    if pd.isna(dt):
        return None

    if isinstance(dt, str):
        dt = dt.strip()

        # Try strict ISO-like formats (YYYY-MM-DD, YYYY/MM/DD, etc.)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                return datetime.strptime(dt, fmt).isoformat()
            except:
                pass

        # Try DD-MM-YYYY only if unambiguous
        for fmt in ("%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(dt, fmt).isoformat()
            except:
                pass

        # Fallback: let pandas decide without dayfirst warning
        try:
            return pd.to_datetime(dt, dayfirst=False).isoformat()
        except:
            return str(dt)

    if isinstance(dt, (pd.Timestamp, datetime)):
        return pd.to_datetime(dt).isoformat()

    return str(dt)

# --------- producer init ----------
def init_producer(broker_uri):
    try:
        p = KafkaProducer(
            bootstrap_servers=broker_uri,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            linger_ms=10
        )
        # quick test send? optional
        return p
    except Exception as e:
        print(f"[WARN] Kafka producer init failed: {e}")
        return None

# --------- send single event ----------
def send_event(producer, topic, event, fallback_dir=None):
    if producer:
        try:
            print('Trying to send to Kafka...')
            producer.send(topic, event)
            producer.flush()
            print(f"Sent to Kafka: station={event.get('power_station')}, id={event.get('id')}")
            return True
        except Exception as e:
            print(f"[ERROR] send to Kafka failed: {e}")
            producer.close()
            return False
    # fallback: write JSON file
    if fallback_dir:
        os.makedirs(fallback_dir, exist_ok=True)
        fname = os.path.join(fallback_dir, f"evt_{int(time.time()*1000)}_{event.get('id','x')}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(event, f, default=str)
        print(f"Wrote fallback event: {fname}")
        return True
    return False

# --------- main flow ----------
def main(csv_path, broker, topic, delay, replay, fallback_dir):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print('read')

    # initialize kafka producer (may be None if broker down)
    producer = init_producer(broker)
    if producer is None:
        print("[WARN] Kafka producer not initialized — will use fallback file writes if enabled.")
        if not fallback_dir:
            print("[WARN] No fallback dir specified; messages will be skipped.")

    loop = True if replay else False

    try:
        while True:
            for idx, row in df.iterrows():
                event = {
                    # prefer outage_date if present otherwise current time
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "state_name": row.get("state_name") or row.get("state_nam",""),
                    "state_code": row.get("state_code", ""),
                    "power_station": row.get("power_station", ""),
                    "monitored_capacity": try_float(row.get("monitored_capacity", "")),
                    "cap_under_outage": try_float(row.get("cap_under_outage", ""))
                }

                ok = send_event(producer, topic, event, fallback_dir=fallback_dir)
                # if kafka failed and fallback not enabled, we skip but continue
                time.sleep(delay)

            if not loop:
                break
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if producer:
            try:
                producer.close()
            except:
                pass

def try_float(x):
    try:
        if x == "" or pd.isna(x):
            return None
        return float(x)
    except Exception:
        # remove commas and try again
        try:
            return float(str(x).replace(",", ""))
        except Exception:
            return None

# -------- CLI ----------
if __name__ == "__main__":
    # no arguments; use defaults
    fallback_dir = "outbox" if WRITE_FALLBACK else None

    main(
        CSV_PATH,
        KAFKA_BROKER,
        TOPIC,
        SLEEP_SECONDS,
        REPLAY,
        fallback_dir
    )
