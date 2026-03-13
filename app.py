import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import datetime
import plotly.express as px
from fpdf import FPDF

# 1. SETUP & THEME
st.set_page_config(page_title="Enterprise Fleet Intelligence", layout="wide")

# AUTHENTICATION (Simple Key)
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.sidebar.title("Secure Login")
        user_pwd = st.sidebar.text_input("Corporate Access Key", type="password")
        if st.sidebar.button("Access Platform"):
            if user_pwd == "fleet2026":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.sidebar.error("Invalid Key")
        return False
    return True

if not check_password():
    st.title("Enterprise Fleet Intelligence")
    st.info("Please login via the sidebar to access AI Analytics.")
    st.stop()

# DARK CAR THEME
car_bg_url = "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=1920"
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("{car_bg_url}");
    background-size: cover; background-attachment: fixed; color: white;
}}
[data-testid="stSidebar"] {{ background-color: rgba(0,0,0,0.85) !important; }}
label, p, .stMarkdown, h1, h2, h3, .stMetric, .stSelectbox {{ color: white !important; }}
div.stButton > button {{
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white; border-radius: 10px; font-weight: bold; width: 100%;
}}
</style>
""", unsafe_allow_html=True)

# 2. RESOURCES & LOGIC
FUEL_PRICE, ANNUAL_MILES = 4.50, 15000
RNN_COLS = ["Model Year", "Make", "Model", "Vehicle Class", "Engine Size", "Cylinders",
            "Transmission", "Fuel Type", "City (L/100km)", "Hwy (L/100km)", "Comb (L/100km)", "CO2 Emissions"]

FUEL_MAP = {"Premium": 0, "Z": 0, "Regular": 1, "X": 1, "Diesel": 2, "D": 2, "Ethanol": 3, "E": 3, "Natural Gas": 4, "N": 4}

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("fuel_efficiency_rnn_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_resources()

def deep_scan_data(df):
    """Scans for Missing, Duplicates, and Outliers (Data Integrity)."""
    report = {
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "outliers": 0
    }
    # Define physical boundary rules for vehicles
    rules = {
        "Engine Size": (0.1, 10.0), 
        "Cylinders": (2, 16), 
        "CO2 Emissions": (20, 1000), 
        "Comb (L/100km)": (1.0, 50.0)
    }
    flagged_idx = []
    for col, (min_v, max_v) in rules.items():
        if col in df.columns:
            mask = (df[col] < min_v) | (df[col] > max_v)
            report["outliers"] += mask.sum()
            flagged_idx.extend(df[mask].index.tolist())
    return report, list(set(flagged_idx))

def nlp_translator(df):
    df.columns = [c.title().replace('_', ' ').strip() for c in df.columns]
    mapping = {"Type Of Fuel": "Fuel Type", "Fueltype": "Fuel Type", "Emissions": "CO2 Emissions", "Combined": "Comb (L/100km)"}
    df = df.rename(columns=mapping)
    if "Transmission" in df.columns:
        df["Trans_Clean"] = df["Transmission"].astype(str).str.upper().str.strip()
        df["Transmission"] = df["Trans_Clean"].apply(lambda x: 2 if x.startswith("CVT") else 1 if x.startswith("M") else 0)
    if "Fuel Type" in df.columns:
        df["Fuel Type"] = df["Fuel Type"].astype(str).str.title().str.strip().map(lambda x: FUEL_MAP.get(x, 1))
    return df

def classify_efficiency(mpg):
    return "Excellent" if mpg > 35 else "Average" if mpg > 20 else "Poor"

def create_pdf(df):
    pdf = FPDF()
    pdf.add_page(); pdf.set_font("helvetica", 'B', 16)
    pdf.cell(200, 10, "Fleet Intelligence Report", ln=True, align='C')
    pdf.set_font("helvetica", size=12); pdf.ln(10)
    pdf.cell(200, 10, f"Avg Efficiency: {df['Predicted_MPG'].mean():.2f} MPG", ln=True)
    return bytes(pdf.output(dest="S"))

# 3. INTERFACE
st.sidebar.title(f"Fleet Intel v5.2")
mode = st.sidebar.radio("Navigation", ["Single Vehicle", "Bulk Fleet Analytics"])

if mode == "Single Vehicle":
    st.header("Vehicle Profile")
    c1, c2 = st.columns(2)
    with c1:
        v_make = st.text_input("Vehicle Make", "Toyota")
        eng = st.number_input("Engine Size (L)", 0.5, 10.0, 2.0)
        cyl = st.number_input("Cylinders", 2, 16, 4)
        fuel_t = st.selectbox("Fuel Type", ["Premium", "Regular", "Diesel", "Ethanol"])
    with c2:
        v_class = st.selectbox("Vehicle Class", ["Truck","Compact", "SUV", "Mid-Size", "Pickup"])
        v_trans = st.selectbox("Transmission", ["Automatic", "Manual", "CVT"])
        co2 = st.number_input("CO2 Emissions (g/km)", 50, 600, 200)
        comb = st.number_input("Combined L/100km", 2.0, 30.0, 9.0)

    if st.button("Generate AI Prediction"):
        f_val = FUEL_MAP.get(fuel_t, 1)
        t_val = 2 if v_trans == "CVT" else 1 if v_trans == "Manual" else 0
        features = np.array([[2026, 0, 0, 0, eng, cyl, t_val, f_val, comb+1, comb-1, comb, co2]])
        scaled = scaler_X.transform(features)
        rnn_in = np.repeat(scaled[:, np.newaxis, :], 5, axis=1) 
        raw_mpg = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in)))[0][0]
        st.metric("Efficiency Score", f"{max(0.1, raw_mpg):.2f} MPG")
        st.success(f"Rating: {classify_efficiency(raw_mpg)}")

else:
    st.header("Enterprise Analytics Engine")
    file = st.file_uploader("Upload Fleet Data", type=["csv", "xlsx"])
    if file:
        df_raw = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df = nlp_translator(df_raw.copy())
        
        # DATA INTEGRITY DASHBOARD
        report, bad_rows = deep_scan_data(df)
        
        with st.expander("Data Integrity Health Report", expanded=True):
            k1, k2, k3 = st.columns(3)
            k1.metric("Missing Cells", report["missing"], delta="Check Data" if report["missing"] > 0 else None, delta_color="inverse")
            k2.metric("Duplicate Rows", report["duplicates"])
            k3.metric("Invalid Outliers", report["outliers"])
            
            if len(bad_rows) > 0:
                st.warning("Some rows contain data that falls outside normal mechanical ranges.")
                if st.checkbox("View flagged rows"):
                    st.dataframe(df_raw.iloc[bad_rows])

        if st.button("Process Intelligence"):
            template = np.zeros((len(df), 12))
            input_df = pd.DataFrame(template, columns=RNN_COLS)
            for col in df.columns:
                if col in RNN_COLS: input_df[col] = df[col]
            
            ai_in_raw = scaler_X.transform(input_df.apply(pd.to_numeric, errors='coerce').fillna(0).values)
            rnn_in = np.repeat(ai_in_raw[:, np.newaxis, :], 5, axis=1)
            raw_preds = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in))).flatten()
            
            df["Predicted_MPG"] = [max(0.1, p) for p in raw_preds]
            df["Annual_Fuel_Cost"] = (ANNUAL_MILES / df["Predicted_MPG"]) * FUEL_PRICE
            df["Efficiency_Rating"] = df["Predicted_MPG"].apply(classify_efficiency)

            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("Total Fleet Spend", f"${df['Annual_Fuel_Cost'].sum():,.0f}")
            m2.metric("Avg Fleet MPG", f"{df['Predicted_MPG'].mean():.1f}")
            st.dataframe(df)
            st.plotly_chart(px.scatter(df, x="Engine Size", y="Predicted_MPG", color="Efficiency_Rating", template="plotly_dark"), use_container_width=True)
            st.download_button("Download Executive PDF", create_pdf(df), "fleet_report.pdf")
