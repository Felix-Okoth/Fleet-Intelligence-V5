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

# DARK THEME (RESTORED)
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

# --- ACTIVITY 1: THE HYBRID INJECTION (CALIBRATED) ---
def apply_hybrid_reality_logic(rnn_mpg, year, make, v_class, fuel_t, engine_size, cylinders, co2):
    """Activity 1: Merges RNN patterns with 6 Ghost Features for 100% Realism."""
    # 1. Weight Evolution
    base_year = 2000
    weight_evolution = 1 - ((year - base_year) * 0.007)
    
    # 2. Manufacturer & Class Mass
    make_bias = {"Toyota": 0.95, "Honda": 0.95, "Ford": 1.10, "Chevrolet": 1.10}
    m_factor = make_bias.get(make, 1.0)
    
    # 3. Stoichiometric Chemistry (The Truth Bridge)
    fuel_chem = {"Regular": 8887, "Premium": 8887, "Diesel": 10180, "Ethanol": 5903}
    energy_constant = fuel_chem.get(fuel_t, 8887)
    chemical_truth_mpg = energy_constant / (max(co2, 1) * 1.609)

    # 4. Thermal Efficiency Ceiling
    friction_loss = (engine_size * 0.09) + (cylinders * 0.05)
    max_physical_cap = (63.0 / (1 + friction_loss)) * (1 / m_factor)

    # --- THE "NOOSE" LOGIC (Variance Detection) ---
    percent_variance = abs(rnn_mpg - chemical_truth_mpg) / chemical_truth_mpg
    
    if percent_variance > 0.15:
        final_mpg = chemical_truth_mpg
    else:
        final_mpg = (rnn_mpg * 0.20) + (chemical_truth_mpg * 0.80)
    
    final_mpg = min(final_mpg, max_physical_cap)
    return round(final_mpg, 2)

# --- DYNAMIC EXECUTIVE PDF ENGINE ---
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    
    # Branding Header
    pdf.set_fill_color(30, 30, 30); pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 24)
    pdf.cell(0, 20, "FLEET STRATEGY & ANALYTICS", ln=True, align='C')
    pdf.set_font("helvetica", '', 10)
    pdf.cell(0, 5, f"REPORT ID: {random.randint(1000,9999)} | ISSUED: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0); pdf.ln(20)

    # Dynamic Insights (Executive Vocabulary)
    openings = ["Strategic Fleet Overview:", "Operational Efficiency Analysis:", "Executive Performance Brief:"]
    insights = [
        "Fleet demonstrates a high-efficiency core with specific cost-containment opportunities.",
        "Sustainable performance metrics indicate significant ROI potential in mid-tier segments.",
        "Analytical mapping identifies key areas for immediate modernization and fuel reduction."
    ]
    
    avg_mpg = df['Predicted_MPG'].mean()
    dist = df['Efficiency_Rating'].value_counts().to_dict()
    
    pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, random.choice(openings), ln=True)
    pdf.set_font("helvetica", '', 11)
    pdf.multi_cell(0, 7, f"{random.choice(insights)} Current fleet average sits at {avg_mpg:.1f} MPG.")
    pdf.ln(5)

    # KPI Summary Grid
    pdf.set_font("helvetica", 'B', 11); pdf.set_fill_color(245, 245, 245)
    pdf.cell(63, 15, f"EXCELLENT: {dist.get('Excellent', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 15, f"AVERAGE: {dist.get('Average', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 15, f"POOR: {dist.get('Poor', 0)}", border=1, ln=True, align='C', fill=True)
    pdf.ln(10)

    # Highlight Table (Top 5)
    pdf.set_font("helvetica", 'B', 12); pdf.cell(0, 10, "High-Impact Asset Highlights", ln=True)
    pdf.set_font("helvetica", 'B', 10); pdf.set_fill_color(0, 114, 255); pdf.set_text_color(2
