import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
import json
import sys

st.set_page_config(page_title="India Power Analysis Dashboard", layout="wide")

os.environ['HADOOP_HOME'] = "C:\\hadoop"
if "C:\\hadoop\\bin" not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + "C:\\hadoop\\bin"

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
spark_available = True

# PATHS (Adjusted to your output)
BASE_DRIVE = r"D:\IISc_Mtech\projects\DEAS\data\output\output_new"

PATH_MASTER_MONTHLY = os.path.join(BASE_DRIVE, "master_tables", "master_monthly_state.csv")
PATH_MASTER_YEARLY  = os.path.join(BASE_DRIVE, "master_tables", "master_yearly_state.csv")
PATH_ASSETS = os.path.join(BASE_DRIVE, "dashboard_assets")

# New: Path for the forecast file
PATH_FORECAST = os.path.join(BASE_DRIVE, "state_monthly_forecast.csv")

@st.cache_resource
def get_spark():
    """Initialize Spark safely. Returns None if it crashes."""
    if not spark_available: return None
    try:
        return SparkSession.builder.appName("Dashboard").master("local[*]").config("spark.driver.memory", "2g").getOrCreate()
    except Exception as e:
        return None

@st.cache_resource
def load_model(_spark):
    """Load Deficit Risk Model."""
    if _spark is None: return None
    path = os.path.join(PATH_ASSETS, "model_a_deficit")
    if os.path.exists(path):
        try: return PipelineModel.load(path)
        except: pass
    return None

@st.cache_data
def load_data():
    # A. Master Monthly
    if not os.path.exists(PATH_MASTER_MONTHLY): 
        df_m = pd.DataFrame()
    else:
        df_m = pd.read_csv(PATH_MASTER_MONTHLY)
        
        # Fix Date
        for c in ['month_start_date', 'join_date', 'date']:
            if c in df_m.columns:
                df_m['date'] = pd.to_datetime(df_m[c], errors='coerce')
                break
        df_m['year'] = df_m['date'].dt.year
    
    # B. Master Yearly
    df_y = pd.read_csv(PATH_MASTER_YEARLY) if os.path.exists(PATH_MASTER_YEARLY) else pd.DataFrame()

    # C. Strategic Forecasts
    path_year_fc = os.path.join(PATH_ASSETS, "yearly_forecasts.csv")
    df_year_fc = pd.read_csv(path_year_fc) if os.path.exists(path_year_fc) else pd.DataFrame()

    # D. Features
    path_feat = os.path.join(PATH_ASSETS, "features_a.json")
    feats = json.load(open(path_feat)) if os.path.exists(path_feat) else []
    
    # E. NEW: Load forecast data
    if os.path.exists(PATH_FORECAST):
        df_forecast = pd.read_csv(PATH_FORECAST)
        df_forecast['month_date'] = pd.to_datetime(df_forecast['month'])
    else:
        df_forecast = pd.DataFrame()
    
    return df_m, df_y, df_year_fc, feats, df_forecast

# Load Data First (Doesn't need Spark)
df_m, df_y, df_year_fc, feature_names, df_forecast = load_data()

# Initialize Spark (Allow failure)
spark = get_spark()
model = load_model(spark)

# Check if we have either historical or forecast data
if (df_m is None or df_m.empty) and df_forecast.empty:
    st.error("Critical Data Missing. Please check paths.")
    st.stop()

st.sidebar.header("Energy Controls")

# Get state list from available data
if not df_m.empty:
    state_list = sorted([str(x) for x in df_m['state_name'].unique()])
elif not df_forecast.empty:
    state_list = sorted([str(x) for x in df_forecast['state_name'].unique()])
else:
    state_list = []

if state_list:
    # Try to find Maharashtra or use first state
    default_ix = state_list.index("maharashtra") if "maharashtra" in [s.lower() for s in state_list] else 0
    sel_state = st.sidebar.selectbox("Select Region", state_list, index=default_ix)
else:
    st.error("No states found in data")
    st.stop()

# Filter data for selected state
if not df_m.empty:
    state_m = df_m[df_m['state_name'] == sel_state].sort_values("date")
    state_y = df_y[df_y['state_name'] == sel_state].sort_values("year") if not df_y.empty else pd.DataFrame()
else:
    state_m = pd.DataFrame()
    state_y = pd.DataFrame()

if not df_forecast.empty:
    state_forecast = df_forecast[df_forecast['state_name'].str.lower() == sel_state.lower()].sort_values("month_date")
else:
    state_forecast = pd.DataFrame()

st.title(f"🇮🇳 Energy Strategy: {sel_state.title()}")

def metric_chart(col, title, series_col, current_val, suffix="", color="blue"):
    with col:
        st.metric(title, f"{current_val:,.1f}{suffix}")
        plot_data = state_m[state_m[series_col] > 0] if not state_m.empty else pd.DataFrame()
        if not plot_data.empty:
            fig = px.line(plot_data, x='date', y=series_col)
            fig.update_traces(line_color=color)
            fig.update_layout(
                showlegend=False, 
                height=50, 
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False, range=[0, plot_data[series_col].max()*1.1])
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

k1, k2, k3, k4 = st.columns(4)

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Supply & Mix", "Strategic Coal", "Grid Reliability", "Forecast Trends", "Long-term Projections", "Simulator"
])

# --- TAB 1: SUPPLY & MIX (Updated Capacity Filter) ---
with tab1:
    if not state_m.empty:
        c1, c2 = st.columns([1, 2])
        
        # A. Capacity Mix (Pie Chart)
        with c1:
            st.subheader("Installed Capacity Mix")
            
            # Year Filter for Capacity
            years = sorted(state_m[state_m['total_capacity_mw'] > 0]['year'].unique(), reverse=True) if 'total_capacity_mw' in state_m.columns else []
            if years:
                sel_year_cap = st.selectbox("Select Year", years, key='cap_year')
                
                # Get Average Capacity for that year
                y_cap = state_m[state_m['year'] == sel_year_cap].mean(numeric_only=True)
                
                if not pd.isna(y_cap.get('total_capacity_mw')):
                    # Define Components
                    cap_vals = {
                        'Coal': y_cap.get('cap_coal_mw', 0),
                        'Gas': y_cap.get('cap_gas_mw', 0),
                        'Hydro': y_cap.get('cap_hydro_mw', 0),
                        'Nuclear': y_cap.get('cap_nuclear_mw', 0),
                        'Renewable (RES)': y_cap.get('cap_res_mw', 0)
                    }
                    # Filter > 0
                    cap_data = {k: v for k, v in cap_vals.items() if v > 0}
                    
                    if cap_data:
                        fig_pie = px.pie(
                            names=list(cap_data.keys()), values=list(cap_data.values()),
                            hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism
                        )
                        fig_pie.update_layout(
                            title=f"Capacity Breakdown ({sel_year_cap})",
                            height=350, margin=dict(t=30, b=0, l=0, r=0)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("No capacity data for selected year.")
            else:
                st.warning("No capacity data available.")

        # B. Generation Trend
        with c2:
            st.subheader("Supply vs Demand Trend")
            fig = go.Figure()
            if 'energy_requirement' in state_m.columns:
                fig.add_trace(go.Scatter(x=state_m['date'], y=state_m.get('energy_requirement',0), 
                                       name='Requirement', fill='tozeroy'))
            if 'total_gen_actual_mu' in state_m.columns or 'total_renew_gen_mu' in state_m.columns:
                tot_sup = state_m.get('total_gen_actual_mu', 0) + state_m.get('total_renew_gen_mu', 0)
                fig.add_trace(go.Scatter(x=state_m['date'], y=tot_sup, 
                                       name='Total Supply', line=dict(color='orange', width=2)))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data available for this state. Check Forecast tabs for predictions.")

# --- TAB 2: STRATEGIC COAL ---
with tab2:
    st.subheader("Strategic Outlook (2024-2026)")
    
    if not df_year_fc.empty:
        y_fc = df_year_fc[df_year_fc['state_name'].str.lower() == sel_state.lower()]
        
        if not y_fc.empty:
            fig_strat = go.Figure()
            # Column Mapping (Handling V3 vs Final names)
            prod_col = 'pred_coal_production_mt' if 'pred_coal_production_mt' in y_fc.columns else 'pred_production_mt'
            cons_col = 'pred_coal_consumption_power_mt' if 'pred_coal_consumption_power_mt' in y_fc.columns else 'pred_consumption_power_mt'
            
            if prod_col in y_fc.columns:
                fig_strat.add_trace(go.Bar(x=y_fc['year'], y=y_fc[prod_col], name='Production Forecast', marker_color='green'))
            if cons_col in y_fc.columns:
                fig_strat.add_trace(go.Bar(x=y_fc['year'], y=y_fc[cons_col], name='Consumption Forecast', marker_color='red'))
            
            fig_strat.update_layout(barmode='group', title="Projected Coal Balance")
            st.plotly_chart(fig_strat, use_container_width=True)
        else:
            st.warning("No strategic forecast for this state.")
    else:
        st.warning("Strategic forecast data missing.")

# --- TAB 3: RELIABILITY ---
with tab3:
    if not state_m.empty:
        c_stk, c_out = st.columns(2)
        
        with c_stk:
            st.markdown("**Coal Stock Composition**")
            # Indigenous vs Import Stack
            fig_stk = go.Figure()
            if 'avg_indigenous_coal_stock_tonnes' in state_m.columns:
                fig_stk.add_trace(go.Scatter(x=state_m['date'], y=state_m['avg_indigenous_coal_stock_tonnes'], 
                                            name='Indigenous', stackgroup='one'))
            if 'avg_import_coal_stock_tonnes' in state_m.columns:
                fig_stk.add_trace(go.Scatter(x=state_m['date'], y=state_m['avg_import_coal_stock_tonnes'], 
                                            name='Imported', stackgroup='one'))
            st.plotly_chart(fig_stk, use_container_width=True)
            
        with c_out:
            st.markdown("**Outage Composition**")
            fig_out = go.Figure()
            if 'avg_total_outage_mw' in state_m.columns:
                fig_out.add_trace(go.Scatter(x=state_m['date'], y=state_m.get('avg_total_outage_mw',0), 
                                            name='Total Outage', fill='tozeroy', line=dict(color='gray')))
            if 'avg_fresh_outage_mw' in state_m.columns:
                fig_out.add_trace(go.Scatter(x=state_m['date'], y=state_m.get('avg_fresh_outage_mw',0), 
                                            name='Fresh Outage', line=dict(color='red')))
            st.plotly_chart(fig_out, use_container_width=True)
    else:
        st.info("No historical reliability data available for this state.")

# --- TAB 4: FORECAST TRENDS (NEW) ---
with tab4:
    if not state_forecast.empty:
        st.subheader("Monthly Forecast Trends (2025-2050)")
        
        # Year range selector
        min_year = int(state_forecast['year'].min())
        max_year = int(state_forecast['year'].max())
        
        year_range = st.slider("Select Year Range", min_year, max_year, (min_year, min(min_year+5, max_year)))
        
        # Filter data
        filtered_fc = state_forecast[(state_forecast['year'] >= year_range[0]) & 
                                     (state_forecast['year'] <= year_range[1])]
        
        # Energy Requirement Forecast
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=filtered_fc['month_date'], 
            y=filtered_fc['predicted_energy_requirement'],
            name='Energy Requirement',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        fig1.update_layout(
            title="Predicted Energy Requirement (MU)",
            xaxis_title="Month",
            yaxis_title="Energy (MU)",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Generation Mix Forecast
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Non-Power Generation Forecast**")
            fig2 = px.line(filtered_fc, x='month_date', y='predicted_non_power_gen')
            fig2.update_traces(line_color='green')
            fig2.update_layout(
                xaxis_title="Month",
                yaxis_title="Generation (MU)",
                height=300
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.markdown("**Renewable Generation Forecast**")
            fig3 = px.line(filtered_fc, x='month_date', y='predicted_total_renew_gen')
            fig3.update_traces(line_color='orange')
            fig3.update_layout(
                xaxis_title="Month",
                yaxis_title="Generation (MU)",
                height=300
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Stacked area chart for total generation
        st.markdown("**Total Generation Mix Forecast**")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=filtered_fc['month_date'],
            y=filtered_fc['predicted_non_power_gen'],
            name='Non-Power Gen',
            stackgroup='one',
            fillcolor='rgba(0, 128, 0, 0.5)'
        ))
        fig4.add_trace(go.Scatter(
            x=filtered_fc['month_date'],
            y=filtered_fc['predicted_total_renew_gen'],
            name='Renewable Gen',
            stackgroup='one',
            fillcolor='rgba(255, 165, 0, 0.5)'
        ))
        fig4.update_layout(
            xaxis_title="Month",
            yaxis_title="Generation (MU)",
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)
        
    else:
        st.warning("No forecast data available.")

# --- TAB 5: LONG-TERM PROJECTIONS (NEW) ---
with tab5:
    if not state_forecast.empty:
        st.subheader("Annual Projections (2025-2050)")
        
        # Aggregate by year
        yearly_agg = state_forecast.groupby('year').agg({
            'predicted_energy_requirement': 'sum',
            'predicted_non_power_gen': 'sum',
            'predicted_total_renew_gen': 'sum'
        }).reset_index()
        
        # Calculate total generation
        yearly_agg['total_generation'] = (yearly_agg['predicted_non_power_gen'] + 
                                         yearly_agg['predicted_total_renew_gen'])
        yearly_agg['supply_demand_gap'] = (yearly_agg['total_generation'] - 
                                          yearly_agg['predicted_energy_requirement'])
        
        # Long-term trend
        fig_long = go.Figure()
        fig_long.add_trace(go.Scatter(
            x=yearly_agg['year'],
            y=yearly_agg['predicted_energy_requirement'],
            name='Energy Requirement',
            line=dict(color='red', width=3)
        ))
        fig_long.add_trace(go.Scatter(
            x=yearly_agg['year'],
            y=yearly_agg['total_generation'],
            name='Total Generation',
            line=dict(color='green', width=3)
        ))
        fig_long.update_layout(
            title="Annual Energy Requirement vs Generation",
            xaxis_title="Year",
            yaxis_title="Energy (MU)",
            height=500
        )
        st.plotly_chart(fig_long, use_container_width=True)
        
        # Supply-Demand Gap
        st.markdown("**Supply-Demand Gap Analysis**")
        fig_gap = px.bar(yearly_agg, x='year', y='supply_demand_gap',
                        color='supply_demand_gap',
                        color_continuous_scale=['red', 'yellow', 'green'])
        fig_gap.update_layout(
            xaxis_title="Year",
            yaxis_title="Gap (MU)",
            height=400
        )
        st.plotly_chart(fig_gap, use_container_width=True)
        
        # Renewable penetration
        yearly_agg['renewable_share'] = (yearly_agg['predicted_total_renew_gen'] / 
                                        yearly_agg['total_generation'] * 100)
        
        st.markdown("**Renewable Energy Penetration**")
        fig_ren = px.line(yearly_agg, x='year', y='renewable_share')
        fig_ren.update_traces(line_color='orange', line_width=3)
        fig_ren.update_layout(
            xaxis_title="Year",
            yaxis_title="Renewable Share (%)",
            height=400
        )
        st.plotly_chart(fig_ren, use_container_width=True)
        
        # Statistics table
        st.markdown("**Key Statistics**")
        stats_df = pd.DataFrame({
            'Metric': ['Avg Energy Requirement (MU/year)', 
                      'Avg Total Generation (MU/year)',
                      'Avg Renewable Share (%)',
                      'Growth Rate (Energy Req.)',
                      'Growth Rate (Total Gen.)'],
            'Value': [
                f"{yearly_agg['predicted_energy_requirement'].mean():,.0f}",
                f"{yearly_agg['total_generation'].mean():,.0f}",
                f"{yearly_agg['renewable_share'].mean():.1f}",
                f"{((yearly_agg['predicted_energy_requirement'].iloc[-1] / yearly_agg['predicted_energy_requirement'].iloc[0]) ** (1/len(yearly_agg)) - 1) * 100:.2f}%",
                f"{((yearly_agg['total_generation'].iloc[-1] / yearly_agg['total_generation'].iloc[0]) ** (1/len(yearly_agg)) - 1) * 100:.2f}%"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
    else:
        st.warning("No forecast data available for long-term projections.")

# --- TAB 6: SIMULATOR (Renamed from original tab4) ---
with tab6:
    st.subheader("Energy Scenario Simulator")
    st.info("This tab can be used for what-if analysis and scenario modeling.")
    
    if not state_forecast.empty:
        # Simple scenario builder
        st.markdown("**Adjust Forecast Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            demand_multiplier = st.slider("Demand Growth Multiplier", 0.5, 2.0, 1.0, 0.1)
            renewable_multiplier = st.slider("Renewable Capacity Multiplier", 0.5, 3.0, 1.0, 0.1)
        
        with col2:
            start_year_sim = st.selectbox("Start Year", sorted(state_forecast['year'].unique()))
            end_year_sim = st.selectbox("End Year", sorted(state_forecast['year'].unique(), reverse=True))
        
        # Apply multipliers
        sim_data = state_forecast[
            (state_forecast['year'] >= start_year_sim) & 
            (state_forecast['year'] <= end_year_sim)
        ].copy()
        
        sim_data['adjusted_demand'] = sim_data['predicted_energy_requirement'] * demand_multiplier
        sim_data['adjusted_renewable'] = sim_data['predicted_total_renew_gen'] * renewable_multiplier
        sim_data['adjusted_total_gen'] = (sim_data['predicted_non_power_gen'] + 
                                          sim_data['adjusted_renewable'])
        
        # Visualize scenario
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sim_data['month_date'],
            y=sim_data['adjusted_demand'],
            name='Adjusted Demand',
            line=dict(color='red', dash='dash')
        ))
        fig_sim.add_trace(go.Scatter(
            x=sim_data['month_date'],
            y=sim_data['adjusted_total_gen'],
            name='Adjusted Generation',
            line=dict(color='green', dash='dash')
        ))
        fig_sim.update_layout(
            title="Scenario Analysis",
            xaxis_title="Month",
            yaxis_title="Energy (MU)",
            height=500
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
    else:
        st.warning("No forecast data available for simulation.")
