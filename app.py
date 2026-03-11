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

# AUTHENTICATION
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

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("fuel_efficiency_rnn_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_resources()

def nlp_translator(df):
    df.columns = [c.title().replace('_', ' ').strip() for c in df.columns]
    mapping = {
        "Type Of Fuel": "Fuel Type", "Fueltype": "Fuel Type", 
        "Emissions": "CO2 Emissions", "Co2 Emissions": "CO2 Emissions",
        "Combined": "Comb (L/100km)", "Combined L/100Km": "Comb (L/100km)"
    }
    df = df.rename(columns=mapping)
    if "Fuel Type" in df.columns:
        fuel_map = {"Z": 0, "X": 1, "D": 2, "E": 3, "N": 4}
        df["Fuel Type"] = df["Fuel Type"].astype(str).str[0].str.upper().map(fuel_map).fillna(1)
    return df

def classify_efficiency(mpg):
    return "Excellent" if mpg > 35 else "Average" if mpg > 20 else "Poor"

def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(200, 10, "Fleet Intelligence Report", ln=True, align='C')
    pdf.set_font("helvetica", size=12); pdf.ln(10)
    pdf.cell(200, 10, f"Avg Efficiency: {df['Predicted_MPG'].mean():.2f} MPG", ln=True)
    if 'Annual_Fuel_Cost' in df.columns:
        pdf.cell(200, 10, f"Total Annual Fleet Cost: ${df['Annual_Fuel_Cost'].sum():,.2f}", ln=True)
    # Correct Byte-output for robust PDF generation
    return bytes(pdf.output(dest="S"))

# 3. INTERFACE
st.sidebar.title("Fleet Intel v5.0")
mode = st.sidebar.radio("Navigation", ["Single Vehicle", "Bulk Fleet Analytics"])

if mode == "Single Vehicle":
    st.header("Vehicle Profile")
    c1, c2 = st.columns(2)
    
    with c1:
        # Added to balance UI and provide context
        v_make = st.text_input("Vehicle Make", "Toyota")
        eng = st.number_input("Engine Size (L)", 0.5, 10.0, 2.0)
        cyl = st.number_input("Cylinders", 2, 16, 4)
        fuel_t = st.selectbox("Fuel Type", ["Premium", "Regular", "Diesel", "Ethanol"])
        
    with c2:
        # Added to balance UI and fill model slots
        v_class = st.selectbox("Vehicle Class", ["Compact", "SUV", "Mid-Size", "Pickup"])
        v_trans = st.selectbox("Transmission", ["Automatic", "Manual"])
        co2 = st.number_input("CO2 Emissions (g/km)", 50, 600, 200)
        comb = st.number_input("Combined L/100km", 2.0, 30.0, 9.0)
        # Placeholder for visual spacing if needed
        st.write("") 

    if st.button("Generate AI Prediction"):
        f_val = {"Premium": 0, "Regular": 1, "Diesel": 2, "Ethanol": 3}.get(fuel_t, 1)
        
        # FEATURE ALIGNMENT MAINTAINED:
        # Slots 0-3 (Year, Make, Model, Class) and Slot 6 (Transmission) 
        # are kept as 0/2026 to avoid breaking the brain's expected input shape.
        features = np.array([[2026, 0, 0, 0, eng, cyl, 0, f_val, comb+1, comb-1, comb, co2]])
        
        scaled = scaler_X.transform(features)
        rnn_in = np.repeat(scaled[:, np.newaxis, :], 5, axis=1) # 5-timestep expansion
        raw_mpg = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in)))[0][0]
        
        rating = classify_efficiency(raw_mpg)
        st.metric("Efficiency Score", f"{raw_mpg:.2f} MPG")
        st.success(f"Rating: {rating}")

else:
    st.header("Enterprise Analytics Engine")
    file = st.file_uploader("Upload Fleet Data", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if st.button("Process Intelligence"):
            df = nlp_translator(df)
            ai_ready_df = df.reindex(columns=RNN_COLS, fill_value=0)
            for col in RNN_COLS:
                ai_ready_df[col] = pd.to_numeric(ai_ready_df[col], errors='coerce').fillna(0)
            
            ai_in = scaler_X.transform(ai_ready_df.values)
            rnn_in = np.repeat(ai_in[:, np.newaxis, :], 5, axis=1)
            raw_preds = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in))).flatten()
            
            df["Predicted_MPG"] = raw_preds
            df["Annual_Fuel_Cost"] = (ANNUAL_MILES / df["Predicted_MPG"]) * FUEL_PRICE
            df["Efficiency_Rating"] = df["Predicted_MPG"].apply(classify_efficiency)

            k1, k2 = st.columns(2)
            k1.metric("Total Fleet Spend", f"${df['Annual_Fuel_Cost'].sum():,.0f}")
            k2.metric("Avg Fleet MPG", f"{df['Predicted_MPG'].mean():.1f}")
            st.dataframe(df)

            fig = px.scatter(df, x="Engine Size", y="Predicted_MPG", color="Efficiency_Rating", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download Executive PDF", create_pdf(df), "fleet_report.pdf")
