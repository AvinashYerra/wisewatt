# WiseWatt — Predict. Detect. Act.

WiseWatt is an AI-powered decision-support system for smart meter intelligence, designed to help electricity distribution companies proactively manage demand and detect anomalies.

---

## What It Does

- Demand Forecasting — Predicts short-term electricity demand at zone level  
- Risk Detection — Identifies high-risk zones prone to grid stress  
- Anomaly Detection — Flags irregular consumption (spikes, drops, tampering patterns)  
- Geospatial Dashboard — Visualizes zones with risk levels on a map  
- Explainability (SHAP) — Explains model predictions for transparency  

---

## Data Note

Due to the lack of publicly available smart meter APIs, this project uses **synthetic data** that mimics real-world electricity consumption patterns, including seasonality, peaks, and anomalies.

---

## Tech Stack

- Python (Pandas, NumPy)  
- XGBoost (Forecasting)  
- Isolation-based logic (Anomaly detection)  
- SHAP (Explainability)  
- Streamlit + PyDeck (Dashboard and Map)  

---

## How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd wisewatt
```
### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. (Optional) Generate data
```bash
python simulation.py
```
This will create:
data/forecast_output.parquet
data/risk_output.parquet
data/shap_summary.png

### 5. Run the application
```bash
streamlit run app.py
```

Open in browser:
```bash
http://localhost:8501
```
---

Live Demo : https://wisewatt.streamlit.app/

---
Impact
 - Reduces unexpected outages
 - Detects potential electricity theft
 - Enables faster, data-driven decisions
