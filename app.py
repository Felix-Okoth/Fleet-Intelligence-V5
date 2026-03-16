import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import datetime
import random
from fpdf import FPDF
import io
import sqlite3
from cryptography.fernet import Fernet 

# =================================================================
# 1. DATABASE & SECURITY ARCHITECTURE (THE FOUNDATION)
# =================================================================
st.set_page_config(page_title="Enterprise Fleet Intelligence", layout="wide")

# Cryptography setup for sensitive data handling
if not os.path.exists("secret.key"):
    with open("secret.key", "wb") as key_file:
        key_file.write(Fernet.generate_key())

def load_key():
    return open("secret.key", "rb").read()

cipher_suite = Fernet(load_key())

def init_db():
    """Initializes the invisible audit and performance tables."""
    conn = sqlite3.connect("fleet_intelligence.db")
    c = conn.cursor()
    # Audit Ledger: Tracks fleet-wide averages over time
    c.execute('''CREATE TABLE IF NOT EXISTS audit_ledger 
                 (id INTEGER PRIMARY KEY, timestamp TEXT, fleet_avg_mpg REAL, total_assets INTEGER)''')
    # Performance Vault: Tracks the delta between AI and Physics logic
    c.execute('''CREATE TABLE IF NOT EXISTS performance_vault 
                 (id INTEGER PRIMARY KEY, timestamp TEXT, rnn_val REAL, physics_val REAL, variance REAL, source TEXT)''')
    conn.commit()
    conn.close()

init_db()

def log_to_ledger(avg_mpg, asset_count):
    conn = sqlite3.connect("fleet_intelligence.db")
    c = conn.cursor()
    c.execute("INSERT INTO audit_ledger (timestamp, fleet_avg_mpg, total_assets) VALUES (?, ?, ?)",
              (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), avg_mpg, asset_count))
    conn.commit()
    conn.close()

def log_to_performance_vault(rnn, physics, var, source="Bulk"):
    conn = sqlite3.connect("fleet_intelligence.db")
    c = conn.cursor()
    c.execute("INSERT INTO performance_vault (timestamp, rnn_val, physics_val, variance, source) VALUES (?, ?, ?, ?, ?)",
              (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), float(rnn), float(physics), float(var), source))
    conn.commit()
    conn.close()

# =================================================================
# 2. AUTHENTICATION & UI BRANDING
# =================================================================
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

# Professional Dark Mode Injection
car_bg_url = "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=1920"
st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), url("{car_bg_url}"); background-size: cover; background-attachment: fixed; }}
[data-testid="stSidebar"] {{ background-color: rgba(10,10,10,0.9) !important; }}
h1, h2, h3, p, label {{ color: #ffffff !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
.stDataFrame {{ background-color: rgba(255,255,255,0.05); border-radius: 10px; }}
div.stButton > button {{ background: linear-gradient(45deg, #00c6ff, #0072ff); color: white; border: none; padding: 10px 24px; border-radius: 8px; font-weight: bold; width: 100%; transition: 0.3s; }}
div.stButton > button:hover {{ transform: scale(1.02); box-shadow: 0 4px 15px rgba(0,198,255,0.4); }}
</style>
""", unsafe_allow_html=True)

# =================================================================
# 3. ANALYTICS ENGINE: PHYSICS + RNN + STOICHIOMETRY
# =================================================================
RNN_COLS = ["Model Year", "Make", "Model", "Vehicle Class", "Engine Size", "Cylinders", "Transmission", "Fuel Type", "City (L/100km)", "Hwy (L/100km)", "Comb (L/100km)", "CO2 Emissions"]

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("fuel_efficiency_rnn_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_resources()

def apply_hybrid_reality_logic(rnn_mpg, year, fuel_t, engine_size, cylinders, co2, source="Bulk"):
    """
    Core Stoichiometric Constraint Engine.
    Prevents AI hallucinations by anchoring results to chemical reality.
    """
    # Fuel Energy Constants (BTU/Gram Equivalent)
    fuel_chem = {"Regular": 8887, "Premium": 8887, "Diesel": 10180, "Ethanol": 5903}
    energy_constant = fuel_chem.get(fuel_t, 8887)
    
    # Physics Calculation: Carbon balance method
    chemical_truth_mpg = energy_constant / (max(co2, 1) * 1.609)
    
    # Mechanical Floor/Ceiling: Friction & Displacement losses
    friction_loss = (engine_size * 0.12) + (cylinders * 0.06)
    max_physical_cap = (68.0 / (1 + friction_loss))
    
    # Hybrid Integration: If AI deviates > 12%, Physics takes over
    percent_variance = abs(rnn_mpg - chemical_truth_mpg) / chemical_truth_mpg
    log_to_performance_vault(rnn_mpg, chemical_truth_mpg, percent_variance, source)
    
    if percent_variance > 0.12:
        final_mpg = chemical_truth_mpg
    else:
        # Weighted blend (85% Physics / 15% RNN patterns)
        final_mpg = (rnn_mpg * 0.15) + (chemical_truth_mpg * 0.85)
        
    return round(min(final_mpg, max_physical_cap), 2)

def classify_efficiency(mpg):
    if mpg > 35: return "Excellent", "rgba(0, 255, 0, 0.4)" 
    elif mpg > 20: return "Average", "rgba(255, 165, 0, 0.4)" 
    else: return "Poor", "rgba(255, 0, 0, 0.4)" 

# =================================================================
# 4. REPORTING & VISUALIZATION (PDF ENGINE)
# =================================================================
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    # ESG Branding Header
    pdf.set_fill_color(20, 20, 20)
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", 'B', 24)
    pdf.cell(0, 25, "FLEET STRATEGY & ESG AUDIT", ln=True, align='C')
    pdf.set_font("helvetica", 'I', 10)
    pdf.cell(0, 5, f"Report ID: {random.randint(100000, 999999)} | Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    
    pdf.set_text_color(40, 40, 40)
    pdf.ln(25)
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Fleet Summary", ln=True)
    pdf.set_font("helvetica", '', 11)
    avg_mpg = df['Predicted_MPG'].mean()
    pdf.multi_cell(0, 7, f"The analyzed fleet segment demonstrates a mean fuel efficiency of {avg_mpg:.2f} MPG. This performance is benchmarked against stoichiometry-validated models for ESG compliance.")
    
    pdf.ln(10)
    # Table Headers
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("helvetica", 'B', 10)
    pdf.cell(45, 10, "Make", 1, 0, 'C', True)
    pdf.cell(45, 10, "Model Year", 1, 0, 'C', True)
    pdf.cell(45, 10, "Efficiency (MPG)", 1, 0, 'C', True)
    pdf.cell(45, 10, "Rating", 1, 1, 'C', True)
    
    pdf.set_font("helvetica", '', 10)
    for i, row in df.head(20).iterrows():
        pdf.cell(45, 10, str(row.get('Make', 'N/A')), 1)
        pdf.cell(45, 10, str(row.get('Model Year', 'N/A')), 1)
        pdf.cell(45, 10, f"{row['Predicted_MPG']:.2f}", 1)
        pdf.cell(45, 10, str(row.get('Efficiency_Rating', 'N/A')), 1, 1)
        
    return pdf.output(dest='S').encode('latin-1')

# =================================================================
# 5. THE INTERFACE (UX & DECISION MAKING)
# =================================================================
st.sidebar.markdown("### Decision Mode")
mode = st.sidebar.radio("Switch View", ["Single Vehicle Insight", "Fleet-Wide Analytics"])

if mode == "Single Vehicle Insight":
    st.header("Intelligence Deep-Dive")
    c1, c2 = st.columns(2)
    with c1:
        v_make = st.text_input("Manufacturer", "Toyota")
        v_year = st.number_input("Model Year", 1995, 2026, 2024)
        v_fuel = st.selectbox("Energy Source", ["Regular", "Premium", "Diesel", "Ethanol"])
        v_eng = st.number_input("Displacement (L)", 0.5, 10.0, 2.5)
    with c2:
        v_cyl = st.number_input("Cylinders", 2, 16, 4)
        v_co2 = st.number_input("CO2 Output (g/km)", 50, 600, 180)
        v_city = st.number_input("City L/100km", 2.0, 30.0, 9.0)
        v_hwy = st.number_input("Hwy L/100km", 2.0, 30.0, 7.0)

    if st.button("Generate Strategic Prediction"):
        # 1. RNN Input Prep
        comb = (v_city * 0.55) + (v_hwy * 0.45)
        raw_row = pd.DataFrame([{"Model Year": v_year, "Make": v_make, "Engine Size": v_eng, "Cylinders": v_cyl, "Fuel Type": v_fuel, "CO2 Emissions": v_co2, "Comb (L/100km)": comb}])
        
        # 2. Mocking RNN prediction for logic demonstration (assuming scaler maps to template)
        rnn_base_mpg = 28.5 # This is where model.predict() would reside
        
        # 3. Apply the OG Physics Logic
        final_mpg = apply_hybrid_reality_logic(rnn_base_mpg, v_year, v_fuel, v_eng, v_cyl, v_co2, source="Single")
        rating, color = classify_efficiency(final_mpg)
        
        st.markdown(f"""
            <div style="background-color: {color}; padding: 25px; border-radius: 12px; width: 60%; border-left: 8px solid white; margin-top: 20px;">
                <h2 style="color: white; margin: 0;">{final_mpg:.2f} MPG</h2>
                <p style="color: white; font-size: 1.2rem; opacity: 0.9;">Efficiency Rating: {rating}</p>
            </div>
        """, unsafe_allow_html=True)
        
        raw_row["Predicted_MPG"] = final_mpg
        raw_row["Efficiency_Rating"] = rating
        pdf_bytes = create_pdf(raw_row)
        st.download_button("Export ESG Document", data=pdf_bytes, file_name=f"Report_{v_make}.pdf", mime="application/pdf")

elif mode == "Fleet-Wide Analytics":
    st.header("Enterprise Analytics Engine")
    uploaded_file = st.file_uploader("Batch Upload Fleet CSV", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if st.button("Execute Bulk Processing"):
            # Simulation of bulk mapping and logic
            results = []
            for _, row in data.iterrows():
                # Extracting columns (assuming standard naming)
                p = apply_hybrid_reality_logic(25.0, 2024, "Regular", 2.0, 4, 200, source="Bulk")
                results.append(p)
            
            data["Predicted_MPG"] = results
            data["Efficiency_Rating"] = [classify_efficiency(x)[0] for x in results]
            
            log_to_ledger(data["Predicted_MPG"].mean(), len(data))
            st.success(f"Processed {len(data)} Assets.")
            st.dataframe(data)
            
            pdf_bulk = create_pdf(data)
            st.download_button("Download Bulk ESG Audit", data=pdf_bulk, file_name="Fleet_Audit.pdf", mime="application/pdf")
