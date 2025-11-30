# India’s Power Generation, Coal Stock & Outage Analytics  
### **Real-Time Monitoring + Forecasting + Dashboards**

A unified data engineering and analytics project that combines **Spark**, **Kafka real-time streaming**, **machine learning**, **HTML/JS dashboards**, and **Streamlit** to analyze and forecast India’s power generation, coal dependency, renewable integration, and outage behavior.

> This project also includes the supporting domain study documented in:  
> **“India’s Power Generation and Consumption Analysis”** (attached in repo)

---

# Features Overview

## **1. Machine Learning & Forecasting (Spark ML)**  
Spark-based distributed forecasting covering:

- **Demand Forecaster** – Predicts next-month energy requirement  
- **Generation Forecaster** – Predicts thermal generation using lag features  
- **Coal Stock Forecasting** – Log-linear estimators for production/consumption/reserves  
- **Outage / Deficit Risk Predictor** – Random forest predicting deficit risks  

These models help uncover:
- Coal dependency trends  
- Seasonal patterns in generation  
- Renewable compensation potential  
- Deficit risks due to outages/coal shortages  

---

## **2. Data Pipelines (PySpark + Structured Streaming)**  

### **Batch Processing**
- Multi-year dataset ingestion  
- Cleaning, transformations, joins  
- Daily → Monthly → Yearly aggregations  
- Spark SQL + DataFrame API  

### **Real-Time Streaming**
- Kafka topic: `telemetry`  
- Producer sends outage events from CSV  
- Consumer processes live events  
- Spark Structured Streaming jobs available for real-time use (optional)  
- Dashboard auto-refreshes from JSON feed  

---

# **3. Dashboards**

## **A. Real-Time Outage Dashboard (HTML + JavaScript)**  
A lightweight visualization UI that shows real-time outage conditions for power plants across India.

### Features
- Auto-refresh every 3s  
- Persisting state selection (dropdown remembers choice)  
- Categories:
  - 🟥 URGENT (≥ 40% outage)  
  - 🟧 MODERATE (10–39%)  
  - 🟩 SMOOTH (< 10%)  
- Bottle-style outage indicator  
- Horizontal layout  
- “No stations to display” contextual messaging  
- Fully responsive (mobile/desktop)

Data is fetched live from: web/latest_events.json
