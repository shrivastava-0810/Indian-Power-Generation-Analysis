import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import glob, os, time
import altair as alt
import streamlit.components.v1 as components
from datetime import datetime

OUT_BASE = r"D:\data"
stream_PATH = os.path.join(OUT_BASE, "stream_all_by_state")
ALERTS_PATH = os.path.join(OUT_BASE, "alerts")

AUTO_REFRESH_SECONDS_DEFAULT = 2
MAX_FILES_READ = 200
MAX_PLANTS_CHART = 8

FULL_COLS = [
    "id","date","state_name","state_code","power_station_name","sector","utility","mode_of_transport",
    "capacity","daily_requirement","daily_receipt","daily_consumption","req_normative_stock","normative_stock_days",
    "indigenous_stock","import_stock","total_stock","stock_days","plf_prcnt","actual_vs_normative_stock_prcnt",
    "is_critical","remarks","event_ts","kafka_ingest_ts","severity"
]

SEVERITY_COLOR = {"NONE": "#6c6c6c", "CRITICAL": "#ff9800", "SUPER": "#ff4d4d"}

st.set_page_config(layout="wide", page_title="Coal Stocks - Streaming Dashboard")
st.title("Coal Stocks — full-schema streaming dashboard")

refresh_sec = st.sidebar.number_input("Auto-refresh (seconds)", min_value=2, max_value=120, value=AUTO_REFRESH_SECONDS_DEFAULT)
st.sidebar.write("Data directories:")
st.sidebar.write(f"- stream: {stream_PATH}")
st.sidebar.write(f"- Alerts: {ALERTS_PATH}")

# try:
#     from streamlit_autorefresh import st_autorefresh
#     # this will cause the script to rerun every refresh_sec seconds
#     st_autorefresh(interval=refresh_sec * 1000, limit=None, key="auto_ref")
# except Exception:
#     # fallback: inject tiny JS that reloads the page; height must be > 0 to run reliably
#     reload_ms = int(refresh_sec) * 1000
#     reload_js = f"""
#     <script>
#     setTimeout(function() {{
#       const url = new URL(window.location.href);
#       url.searchParams.set("_t", Date.now());
#       window.location.href = url.toString();
#     }}, {reload_ms});
#     </script>
#     """
#     components.html(reload_js, height=1)

def colored_circle_html(color, size=12):
    return f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{color};display:inline-block;"></div>'

def list_states_from_partitions(base_path=stream_PATH):
    if not os.path.exists(base_path):
        return []
    states = []
    for entry in os.listdir(base_path):
        if entry.startswith("state_name="):
            states.append(entry.split("state_name=",1)[1])
    return sorted(list(dict.fromkeys(states)), key=lambda s: s.lower())

def list_parquet_files(folder, limit=MAX_FILES_READ):
    if not os.path.exists(folder):
        return []
    pattern = os.path.join(folder, "*.parquet")
    files = [f for f in glob.glob(pattern) if not f.endswith(".crc") and os.path.getsize(f) > 0]
    files = sorted(files, key=os.path.getmtime, reverse=True)[:limit]
    return files

def read_parquet_files(files):
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    try:
        return pd.concat(dfs, ignore_index=True)
    except Exception:
        # fallback to assembling records
        records = []
        for d in dfs:
            records.extend(d.to_dict(orient="records"))
        return pd.DataFrame.from_records(records)

def read_stream_state(state_name):
    folder = os.path.join(stream_PATH, f"state_name={state_name}")
    files = list_parquet_files(folder)
    return read_parquet_files(files)

def read_alerts_state(state_name=None):
    files = list_parquet_files(ALERTS_PATH)
    df = read_parquet_files(files)
    if df.empty:
        return df
    if state_name and "state_name" in df.columns:
        try:
            df = df[df["state_name"] == state_name]
        except Exception:
            pass
    return df

def ensure_columns(df, wanted_cols):
    for c in wanted_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def coerce_numeric_columns(df, numeric_cols):
    for c in numeric_cols:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
            except Exception:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

state_list = list_states_from_partitions()
if not state_list:
    st.error(f"No state partitions found in: {stream_PATH}\nEnsure the receiver wrote stream data.")
    st.stop()

default_index = 0
for i,s in enumerate(state_list):
    if s.lower() == "maharashtra":
        default_index = i
        break

sel_state = st.sidebar.selectbox("Select state", state_list, index=default_index)

with st.spinner(f"Loading data for {sel_state} ..."):
    stream_df = read_stream_state(sel_state)
    alerts_df = read_alerts_state(sel_state)

stream_df = ensure_columns(stream_df, FULL_COLS)
alerts_df = ensure_columns(alerts_df, FULL_COLS)

numeric_cols = [
    "capacity","daily_requirement","daily_receipt","daily_consumption","req_normative_stock",
    "normative_stock_days","indigenous_stock","import_stock","total_stock","stock_days",
    "plf_prcnt","actual_vs_normative_stock_prcnt"
]
stream_df = coerce_numeric_columns(stream_df, numeric_cols)
alerts_df = coerce_numeric_columns(alerts_df, numeric_cols)

if "event_ts" in stream_df.columns:
    try:
        stream_df["event_ts"] = pd.to_datetime(stream_df["event_ts"], errors="coerce")
    except Exception:
        pass
if "event_ts" in alerts_df.columns:
    try:
        alerts_df["event_ts"] = pd.to_datetime(alerts_df["event_ts"], errors="coerce")
    except Exception:
        pass

c1,c2,c3,c4 = st.columns([2,1,1,1])
num_plants = int(stream_df["power_station_name"].nunique()) if not stream_df.empty else 0
num_alerts = int(len(alerts_df)) if not alerts_df.empty else 0
num_super = int(((alerts_df.get("severity") == "SUPER") | (alerts_df.get("severity") == "super")).sum()) if not alerts_df.empty else 0
c1.metric("Plants seen", num_plants)
c2.metric("Alerts", num_alerts)
c3.metric("Super alerts", num_super)
c4.write("Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

st.markdown("---")

st.subheader("Alerts (flagged rows)")
if alerts_df.empty:
    st.info("No alerts for this state yet.")
else:
    display_alerts = alerts_df.copy()
    display_alerts["severity"] = display_alerts.get("severity", display_alerts.get("is_critical", pd.Series())).fillna("NONE").astype(str).str.upper()
    display_alerts["alert_marker"] = display_alerts["severity"].apply(lambda s: colored_circle_html(SEVERITY_COLOR.get(s, "#6c6c6c")))

    cols_present = ["alert_marker"] + [c for c in FULL_COLS if c in display_alerts.columns]
    display_alerts = display_alerts[cols_present]

    if "event_ts" in display_alerts.columns:
        try:
            display_alerts = display_alerts.sort_values("event_ts", ascending=False)
        except Exception:
            pass

    header_cells = "".join([f"<th style='padding:8px;border:1px solid #e8e8e8;background:#f7f7f7;text-align:left'>{c}</th>" for c in cols_present])
    def row_html(row):
        sev = (row.get("severity") or "NONE").upper()
        bg = "#fff1f0" if sev == "SUPER" else ("#fff8e6" if sev == "CRITICAL" else "white")
        cells = ""
        for c in cols_present:
            val = row.get(c, "")
            if pd.isna(val):
                val = ""
            if c == "alert_marker":
                cells += f"<td style='padding:6px;border:1px solid #eee'>{val}</td>"
            else:
                safe = str(val).replace("<","&lt;").replace(">","&gt;")
                cells += f"<td style='padding:6px;border:1px solid #eee'>{safe}</td>"
        return f"<tr style='background:{bg}'>{cells}</tr>"

    rows_html = "".join(display_alerts.apply(lambda r: row_html(r), axis=1).tolist())
    table_html = f"""
    <div style="max-height:420px; overflow:auto;">
      <table style="border-collapse:collapse; width:100%; font-size:13px; font-family:Arial,Helvetica,sans-serif">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.write(table_html, unsafe_allow_html=True)

st.markdown("---")

st.subheader("All incoming rows (stream) — recent sample")
if stream_df.empty:
    st.info("No stream data yet for this state.")
else:
    stream_display = stream_df.copy()
    if "event_ts" in stream_display.columns:
        try:
            stream_display = stream_display.sort_values("event_ts", ascending=False).head(500)
        except Exception:
            stream_display = stream_display.head(500)
    else:
        stream_display = stream_display.head(500)

    cols_to_show = [c for c in FULL_COLS if c in stream_display.columns]
    st.dataframe(stream_display[cols_to_show], use_container_width=True)

    csv = stream_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download stream sample (CSV)", csv, file_name=f"stream_{sel_state}.csv", mime="text/csv")

st.markdown("---")

# st.subheader("Per-plant total_stock trend (top plants)")
# if stream_df.empty or "power_station_name" not in stream_df.columns or "event_ts" not in stream_df.columns:
#     st.info("Not enough data for per-plant charts (need power_station_name & event_ts).")
# else:
#     plot_df = stream_df.copy()
#     plot_df["event_ts"] = pd.to_datetime(plot_df["event_ts"], errors="coerce")
#     latest = plot_df.sort_values("event_ts", ascending=False).drop_duplicates(subset=["power_station_name"])
#     top_plants = latest["power_station_name"].dropna().tolist()[:MAX_PLANTS_CHART]
#     selected = st.multiselect("Plants to show (top):", top_plants, default=top_plants[:4])
#     if selected:
#         df_plot = plot_df[plot_df["power_station_name"].isin(selected)].copy()
#         if "total_stock" in df_plot.columns:
#             df_plot["total_stock"] = pd.to_numeric(df_plot["total_stock"], errors="coerce")
#             y_field = "total_stock"
#         else:
#             fallback = "daily_consumption" if "daily_consumption" in df_plot.columns else df_plot.columns[0]
#             df_plot[fallback] = pd.to_numeric(df_plot[fallback], errors="coerce")
#             y_field = fallback

#         base = alt.Chart(df_plot).encode(x=alt.X("event_ts:T", title="Time"), color=alt.Color("power_station_name:N", title="Plant"))
#         line = base.mark_line().encode(y=alt.Y(f"{y_field}:Q", title=y_field))
#         points = base.mark_point().encode(tooltip=["power_station_name","event_ts", y_field, "is_critical"])
#         st.altair_chart((line + points).interactive().properties(height=360), use_container_width=True)


st.write("Last refreshed:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
