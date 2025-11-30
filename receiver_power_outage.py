# receiver.py (fixed)
import json
import sys
import os
from kafka import KafkaConsumer
from datetime import datetime, timezone

KAFKA_BROKER = "localhost:9092"
TOPIC = "telemetry"

WRITE_TO_FILE = False
OUT_FILE = "received_events.log"

WEB_DIR = "web"
WEB_JSON = os.path.join(WEB_DIR, "latest_events.json")

def main():
    print(f"[INFO] Starting Kafka consumer on broker={KAFKA_BROKER}, topic={TOPIC}")

    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            group_id="telemetry_reader_v4",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=3000
        )
        print("[INFO] Consumer connected.")
    except Exception as e:
        print(f"[ERROR] Could not connect to Kafka: {e}")
        sys.exit(1)

    out_f = None
    if WRITE_TO_FILE:
        try:
            out_f = open(OUT_FILE, "a", encoding="utf-8")
            print(f"[INFO] Appending received messages to {OUT_FILE}")
        except Exception as e:
            print(f"[WARN] Could not open output file {OUT_FILE}: {e}")
            out_f = None

    stations = {}
    os.makedirs(WEB_DIR, exist_ok=True)

    try:
        while True:
            print("[INFO] Waiting for messages...")
            for msg in consumer:
                event = msg.value
                station = event.get("power_station", "UNKNOWN")

                mc = event.get("monitored_capacity")
                ou = event.get("cap_under_outage")
                try:
                    pct = (float(ou) / float(mc) * 100) if mc not in (None, "", 0, "0") else None
                except Exception:
                    pct = None

                stations[station] = {
                    "ts": event.get("ts"),
                    "state_name": event.get("state_name"),
                    "state_code": event.get("state_code"),
                    "power_station": station,
                    "monitored_capacity": mc,
                    "cap_under_outage": ou,
                    "pct_outage": pct
                }

                payload = {
                    "last_update": event.get("ts"),
                    "stations": stations
                }
                try:
                    tmp_path = WEB_JSON + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as wf:
                        json.dump(payload, wf, ensure_ascii=False, indent=2)
                        wf.flush()
                        os.fsync(wf.fileno())
                    os.replace(tmp_path, WEB_JSON)
                except Exception as e:
                    print(f"[WARN] Could not write dashboard JSON: {e}")

                try:
                    print("\n----- RECEIVED EVENT -----")
                    print(json.dumps(event, indent=2))
                except Exception:
                    print("Received event (could not pretty-print):", event)

                if out_f and (not out_f.closed):
                    try:
                        record = {
                            "received_at": datetime.now(timezone.utc).isoformat(),
                            "event": event
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                    except Exception as e:
                        print(f"[WARN] Failed to write to log file: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"[ERROR] Consumer loop crashed: {e}")
    finally:
        try:
            if out_f and (not out_f.closed):
                out_f.close()
                print(f"[INFO] Closed file {OUT_FILE}")
        except Exception as e:
            print(f"[WARN] Error closing file: {e}")

        try:
            consumer.close()
            print("[INFO] Consumer closed.")
        except Exception as e:
            print(f"[WARN] Error closing consumer: {e}")


if __name__ == "__main__":
    main()