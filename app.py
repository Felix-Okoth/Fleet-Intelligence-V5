import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import datetime
import random
import plotly.express as px
from fpdf import FPDF
import io

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

# DARK THEME
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

def apply_hybrid_reality_logic(rnn_mpg, year, make, v_class, fuel_t, engine_size, cylinders, co2):
    base_year = 2000
    make_bias = {"Toyota": 0.95, "Honda": 0.95, "Ford": 1.10, "Chevrolet": 1.10}
    m_factor = make_bias.get(make, 1.0)
    fuel_chem = {"Regular": 8887, "Premium": 8887, "Diesel": 10180, "Ethanol": 5903}
    energy_constant = fuel_chem.get(fuel_t, 8887)
    chemical_truth_mpg = energy_constant / (max(co2, 1) * 1.609)
    friction_loss = (engine_size * 0.09) + (cylinders * 0.05)
    max_physical_cap = (63.0 / (1 + friction_loss)) * (1 / m_factor)
    
    percent_variance = abs(rnn_mpg - chemical_truth_mpg) / chemical_truth_mpg
    if percent_variance > 0.15:
        final_mpg = chemical_truth_mpg
    else:
        final_mpg = (rnn_mpg * 0.20) + (chemical_truth_mpg * 0.80)
    return round(min(final_mpg, max_physical_cap), 2)

# --- REPAIRED EXECUTIVE PDF ENGINE ---
def create_pdf(df, fig):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header
    pdf.set_fill_color(25, 25, 25); pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 22)
    pdf.cell(0, 20, "EXECUTIVE FLEET INTELLIGENCE", ln=True, align='C')
    pdf.set_font("helvetica", '', 10)
    pdf.cell(0, 5, f"CONFIDENTIAL | REF: {random.randint(1000,9999)} | DATE: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0); pdf.ln(20)

    # 2. Robust Column Finder (Fixes the KeyError)
    co2_col = "CO2 Emissions" if "CO2 Emissions" in df.columns else [c for c in df.columns if "CO2" in c.upper()][0]
    
    total_spend = df['Annual_Fuel_Cost'].sum()
    total_co2_tonnes = (df[co2_col].mean() * ANNUAL_MILES * 1.609 * len(df)) / 1_000_000
    
    pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, "1. Executive Summary & ESG Impact", ln=True)
    pdf.set_font("helvetica", '', 11)
    summary_text = (f"Annual Fleet Expenditure: ${total_spend:,.2f}. "
                    f"Carbon Exposure: {total_co2_tonnes:.1f} Metric Tonnes CO2/yr. "
                    "Data highlights strategic modernization opportunities in low-efficiency clusters.")
    pdf.multi_cell(0, 7, summary_text)
    pdf.ln(5)

    # 3. KPI Grid
    dist = df['Efficiency_Rating'].value_counts().to_dict()
    pdf.set_font("helvetica", 'B', 11); pdf.set_fill_color(242, 242, 242)
    pdf.cell(63, 12, f"EXCELLENT: {dist.get('Excellent', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 12, f"AVERAGE: {dist.get('Average', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 12, f"POOR: {dist.get('Poor', 0)}", border=1, ln=True, align='C', fill=True)
    pdf.ln(10)

    # 4. Image Injection
    fig.write_image("temp_chart.png")
    pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, "2. Efficiency Distribution Analysis", ln=True)
    pdf.image("temp_chart.png", x=10, y=pdf.get_y(), w=190)
    pdf.ln(105)

    # 5. Risk Table
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, "3. Replacement Priority List", ln=True)
    pdf.set_font("helvetica", 'B', 10); pdf.set_fill_color(0, 114, 255); pdf.set_text_color(255, 255, 255)
    
    headers = ["Manufacturer", "Year", "Efficiency", "Action Status"]
    widths = [50, 30, 40, 70]
    for i, h in enumerate(headers): pdf.cell(widths[i], 10, h, border=1, align='C', fill=True)
    pdf.ln(); pdf.set_text_color(0, 0, 0); pdf.set_font("helvetica", '', 10)

    at_risk = df[df['Efficiency_Rating'] == 'Poor'].sort_values(by="Model Year").head(15)
    for _, row in at_risk.iterrows():
        status = "CRITICAL REPLACEMENT" if row['Model Year'] < 2018 else "OPERATIONAL REVIEW"
        pdf.cell(widths[0], 10, str(row['Make']), border=1, align='C')
        pdf.cell(widths[1], 10, str(row['Model Year']), border=1, align='C')
        pdf.cell(widths[2], 10, f"{row['Predicted_MPG']:.1f} MPG", border=1, align='C')
        pdf.cell(widths[3], 10, status, border=1, align='C')
        pdf.ln()

    return bytes(pdf.output())

def nlp_translator(df):
    # Aggressive cleaning of column headers
    df.columns = [c.title().strip().replace('  ', ' ') for c in df.columns]
    mapping = {
        "Type Of Fuel": "Fuel Type", "Fueltype": "Fuel Type", 
        "Emissions": "CO2 Emissions", "Co2 Emissions": "CO2 Emissions",
        "Combined": "Comb (L/100km)", "Comb (L/100Km)": "Comb (L/100km)"
    }
    df = df.rename(columns=mapping)
    if "Transmission" in df.columns:
        df["Transmission"] = df["Transmission"].astype(str).str.upper().apply(lambda x: 2 if "CVT" in x else 1 if "M" in x else 0)
    if "Fuel Type" in df.columns:
        df["Fuel Type"] = df["Fuel Type"].astype(str).str.title().map(lambda x: FUEL_MAP.get(x, 1))
    return df

def prepare_ai_input(df, scaler_X):
    template = np.zeros((len(df), 12))
    input_df = pd.DataFrame(template, columns=RNN_COLS)
    for col in df.columns:
        if col in RNN_COLS: input_df[col] = df[col]
    return scaler_X.transform(input_df.apply(pd.to_numeric, errors='coerce').fillna(0).values)

def classify_efficiency(mpg):
    return "Excellent" if mpg > 35 else "Average" if mpg > 20 else "Poor"

# 3. INTERFACE
st.sidebar.title("Fleet Intel v6.0")
mode = st.sidebar.radio("Navigation", ["Single Vehicle", "Bulk Fleet Analytics"])

if mode == "Single Vehicle":
    st.header("Vehicle Profile Prediction")
    c1, c2 = st.columns(2)
    with c1:
        v_make = st.text_input("Make", "Toyota")
        eng = st.number_input("Engine (L)", 0.5, 10.0, 2.0)
        cyl = st.number_input("Cylinders", 2, 16, 4)
        fuel_t = st.selectbox("Fuel", ["Regular", "Premium", "Diesel", "Ethanol"])
        v_year = st.number_input("Year", 1995, 2026, 2024)
    with c2:
        v_class = st.selectbox("Class", ["Mid-Size", "Compact", "SUV", "Pickup", "Truck"])
        v_trans = st.selectbox("Trans", ["Automatic", "Manual", "CVT"])
        co2 = st.number_input("CO2 (g/km)", 50, 600, 200)
        city_l = st.number_input("City L/100km", 2.0, 30.0, 10.0)
        hwy_l = st.number_input("Hwy L/100km", 2.0, 30.0, 8.0)
        comb = (city_l * 0.55) + (hwy_l * 0.45)

    if st.button("Predict"):
        row = pd.DataFrame([{"Model Year": v_year, "Make": v_make, "Engine Size": eng, "Cylinders": cyl, "Fuel Type": fuel_t, "Vehicle Class": v_class, "Transmission": v_trans, "CO2 Emissions": co2, "City (L/100km)": city_l, "Hwy (L/100km)": hwy_l, "Comb (L/100km)": comb}])
        cleaned = nlp_translator(row)
        rnn_in = np.repeat(prepare_ai_input(cleaned, scaler_X)[:, np.newaxis, :], 5, axis=1)
        raw_mpg = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in)))[0][0]
        final = apply_hybrid_reality_logic(raw_mpg, v_year, v_make, v_class, fuel_t, eng, cyl, co2)
        st.metric("Efficiency Score", f"{final:.2f} MPG")

else:
    st.header("Bulk Analytics Engine")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df_raw = pd.read_csv(file)
        df_processed = nlp_translator(df_raw.copy())
        if st.button("Analyze Fleet"):
            rnn_in = np.repeat(prepare_ai_input(df_processed, scaler_X)[:, np.newaxis, :], 5, axis=1)
            raw_preds = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in))).flatten()
            
            final_mpg = []
            for i, p in enumerate(raw_preds):
                r = df_raw.iloc[i]
                final_mpg.append(apply_hybrid_reality_logic(p, r.get("Model Year", 2024), r.get("Make", "Unknown"), r.get("Vehicle Class", "Mid-Size"), r.get("Fuel Type", "Regular"), r.get("Engine Size", 2.0), r.get("Cylinders", 4), r.get("CO2 Emissions", 200)))

            df_processed["Predicted_MPG"] = final_mpg
            df_processed["Annual_Fuel_Cost"] = (ANNUAL_MILES / df_processed["Predicted_MPG"]) * FUEL_PRICE
            df_processed["Efficiency_Rating"] = df_processed["Predicted_MPG"].apply(classify_efficiency)
            
            st.metric("Total Fleet Spend", f"${df_processed['Annual_Fuel_Cost'].sum():,.0f}")
            st.dataframe(df_processed)
            
            fig = px.scatter(df_processed, x="Engine Size", y="Predicted_MPG", color="Efficiency_Rating", template="plotly_dark", title="Fleet Performance Clustering")
            st.plotly_chart(fig, use_container_width=True)
            
            report_data = create_pdf(df_processed, fig)
            st.download_button("Download Strategy Report", data=report_data, file_name=f"Strategy_Report_{datetime.date.today()}.pdf", mime="application/pdf")
