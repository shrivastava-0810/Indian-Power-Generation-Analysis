import pandas as pd

INP_PATH = r'F:\DataEngineering\Project\LiveProject\daily-power-outage.csv'
OUT_PATH = r'F:\DataEngineering\Project\LiveProject\processed_daily_power_outage.csv'

df = pd.read_csv(INP_PATH)

df = df.drop(columns=['remarks','region','sector','expected_sync_date', 'id', 'station_type', 'power_station_unit', 'outage_type', 'outage_date'])

df.to_csv(OUT_PATH, index=False)