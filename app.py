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
import sqlite3
from cryptography.fernet import Fernet 

# 1. SETUP, THEME & SECURITY INITIALIZATION
st.set_page_config(page_title="Enterprise Fleet Intelligence", layout="wide")

if not os.path.exists("secret.key"):
    with open("secret.key", "wb") as key_file:
        key_file.write(Fernet.generate_key())

def load_key():
    return open("secret.key", "rb").read()

cipher_suite = Fernet(load_key())

def init_db():
    conn = sqlite3.connect("fleet_intelligence.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audit_ledger 
                 (id INTEGER PRIMARY KEY, timestamp TEXT, fleet_avg_mpg REAL, total_assets INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS performance_vault 
                 (id INTEGER PRIMARY KEY, timestamp TEXT, rnn_val REAL, physics_val REAL, variance REAL, source TEXT)''')
    conn.commit()
    conn.close()

init_db()

def sanitize_input(text):
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "SELECT", "--", ";", "<script>"]
    clean_text = str(text)
    for word in forbidden:
        clean_text = clean_text.replace(word, "[PROTECTED]")
    return clean_text

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

# DARK THEME (RESTORED OG STYLE)
car_bg_url = "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=1920"
st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("{car_bg_url}"); background-size: cover; background-attachment: fixed; color: white; }}
[data-testid="stSidebar"] {{ background-color: rgba(0,0,0,0.85) !important; }}
label, p, .stMarkdown, h1, h2, h3, .stMetric, .stSelectbox {{ color: white !important; }}
div.stButton > button {{ background: linear-gradient(to right, #00c6ff, #0072ff); color: white; border-radius: 10px; font-weight: bold; width: 100%; }}
</style>
""", unsafe_allow_html=True)

# 2. RESOURCES & LOGIC
FUEL_PRICE, ANNUAL_MILES = 4.50, 15000
RNN_COLS = ["Model Year", "Make", "Model", "Vehicle Class", "Engine Size", "Cylinders", "Transmission", "Fuel Type", "City (L/100km)", "Hwy (L/100km)", "Comb (L/100km)", "CO2 Emissions"]
FUEL_MAP = {"Premium": 0, "Z": 0, "Regular": 1, "X": 1, "Diesel": 2, "D": 2, "Ethanol": 3, "E": 3, "Natural Gas": 4, "N": 4}

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("fuel_efficiency_rnn_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_resources()

# RESTORED HEURISTIC RULE
def classify_efficiency(mpg):
    if mpg > 35: return "Excellent", "#00FF00" # Green
    elif mpg > 20: return "Average", "#FFA500" # Orange
    else: return "Poor", "#FF0000" # Red

def apply_hybrid_reality_logic(rnn_mpg, year, make, v_class, fuel_t, engine_size, cylinders, co2, source="Bulk"):
    make_bias = {"Toyota": 0.95, "Honda": 0.95, "Ford": 1.10, "Chevrolet": 1.10}
    m_factor = make_bias.get(make, 1.0)
    fuel_chem = {"Regular": 8887, "Premium": 8887, "Diesel": 10180, "Ethanol": 5903}
    energy_constant = fuel_chem.get(fuel_t, 8887)
    
    chemical_truth_mpg = energy_constant / (max(co2, 1) * 1.609)
    friction_loss = (engine_size * 0.12) + (cylinders * 0.06)
    max_physical_cap = (68.0 / (1 + friction_loss)) * (1 / m_factor)
    percent_variance = abs(rnn_mpg - chemical_truth_mpg) / chemical_truth_mpg
    
    log_to_performance_vault(rnn_mpg, chemical_truth_mpg, percent_variance, source)
    
    if percent_variance > 0.12:
        final_mpg = chemical_truth_mpg
    else:
        final_mpg = (rnn_mpg * 0.15) + (chemical_truth_mpg * 0.85)
        
    return round(min(final_mpg, max_physical_cap), 2)

# --- THE ESG-READY PDF ENGINE (OG) ---
def create_pdf(df, fig=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(25, 25, 25); pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 22)
    pdf.cell(0, 20, "FLEET STRATEGY & ESG ANALYTICS", ln=True, align='C')
    pdf.set_font("helvetica", '', 10); pdf.cell(0, 5, f"REF: {random.randint(1000,9999)} | GENERATED: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0); pdf.ln(15)
    
    pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, "Strategic Overview:", ln=True)
    pdf.set_font("helvetica", '', 11); avg_mpg = df['Predicted_MPG'].mean()
    pdf.multi_cell(0, 7, f"The fleet trajectory indicates a healthy high-efficiency core with an average of {avg_mpg:.1f} MPG."); pdf.ln(5)

    dist = df['Efficiency_Rating'].value_counts().to_dict()
    pdf.set_font("helvetica", 'B', 11); pdf.set_fill_color(242, 242, 242)
    pdf.cell(63, 15, f"EXCELLENT: {dist.get('Excellent', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 15, f"AVERAGE: {dist.get('Average', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 15, f"POOR: {dist.get('Poor', 0)}", border=1, ln=True, align='C', fill=True); pdf.ln(10)

    pdf_out = pdf.output()
    return bytes(pdf_out) if not isinstance(pdf_out, str) else pdf_out.encode('latin-1')

def nlp_translator(df):
    df.columns = [sanitize_input(c.title().replace('_', ' ').strip()) for c in df.columns]
    mapping = {"Type Of Fuel": "Fuel Type", "Fueltype": "Fuel Type", "Emissions": "CO2 Emissions", "Co2 Emissions": "CO2 Emissions", "Combined": "Comb (L/100km)"}
    df = df.rename(columns=mapping)
    return df

def prepare_ai_input(df, scaler_X):
    template = np.zeros((len(df), 12))
    input_df = pd.DataFrame(template, columns=RNN_COLS)
    for col in df.columns:
        if col in RNN_COLS: input_df[col] = df[col]
    return scaler_X.transform(input_df.apply(pd.to_numeric, errors='coerce').fillna(0).values)

# 3. INTERFACE
st.sidebar.title(f"Fleet Intel v6.5")
mode = st.sidebar.radio("Navigation", ["Single Vehicle", "Bulk Fleet Analytics", "Audit History"])

if mode == "Single Vehicle":
    st.header("Vehicle Profile")
    c1, c2 = st.columns(2)
    with c1:
        v_make = sanitize_input(st.text_input("Vehicle Make", "Toyota"))
        eng = st.number_input("Engine Size (L)", 0.5, 10.0, 2.0)
        cyl = st.number_input("Cylinders", 2, 16, 4)
        fuel_t = st.selectbox("Fuel Type", ["Regular", "Premium", "Diesel", "Ethanol"])
        v_year = st.number_input("Model Year", 1995, 2026, 2024)
    with c2:
        v_class = st.selectbox("Vehicle Class", ["Mid-Size", "Compact", "SUV", "Pickup", "Truck"])
        v_trans = st.selectbox("Transmission", ["Automatic", "Manual", "CVT"])
        co2 = st.number_input("CO2 Emissions (g/km)", 50, 600, 200)
        city_l = st.number_input("City (L/100km)", 2.0, 30.0, 10.0)
        hwy_l = st.number_input("Hwy (L/100km)", 2.0, 30.0, 8.0)
        comb = (city_l * 0.55) + (hwy_l * 0.45)

    if st.button("Generate AI Prediction"):
        single_row = pd.DataFrame([{"Model Year": v_year, "Make": v_make, "Engine Size": eng, "Cylinders": cyl, "Fuel Type": fuel_t, "Vehicle Class": v_class, "Transmission": v_trans, "CO2 Emissions": co2, "City (L/100km)": city_l, "Hwy (L/100km)": hwy_l, "Comb (L/100km)": comb}])
        cleaned_df = nlp_translator(single_row)
        ai_in_raw = prepare_ai_input(cleaned_df, scaler_X)
        rnn_in = np.repeat(ai_in_raw[:, np.newaxis, :], 5, axis=1) 
        raw_mpg = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in)))[0][0]
        display_mpg = apply_hybrid_reality_logic(raw_mpg, v_year, v_make, v_class, fuel_t, eng, cyl, co2, source="Single")
        
        # RESTORED UI ELEMENTS
        rating, color = classify_efficiency(display_mpg)
        st.markdown(f"**{v_year} Efficiency Score**")
        st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">
                <h2 style="color: black; margin: 0;">{display_mpg:.2f} MPG - {rating}</h2>
            </div>
        """, unsafe_allow_html=True)

elif mode == "Bulk Fleet Analytics":
    st.header("Enterprise Analytics Engine")
    file = st.file_uploader("Upload Fleet Data", type=["csv", "xlsx"])
    if file:
        df_raw = pd.read_csv(file) if file.name.lower().endswith('.csv') else pd.read_excel(file)
        df_processed = nlp_translator(df_raw.copy())
        
        if st.button("Process Intelligence"):
            ai_in_raw = prepare_ai_input(df_processed, scaler_X)
            rnn_in = np.repeat(ai_in_raw[:, np.newaxis, :], 5, axis=1)
            raw_preds = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in))).flatten()
            
            final_mpg = []
            for i, p in enumerate(raw_preds):
                row = df_raw.iloc[i] 
                real_p = apply_hybrid_reality_logic(p, row.get("Model Year", 2024), row.get("Make", "Unknown"), row.get("Vehicle Class", "Mid-Size"), row.get("Fuel Type", "Regular"), row.get("Engine Size", 2.0), row.get("Cylinders", 4), row.get("CO2 Emissions", 200), source="Bulk")
                final_mpg.append(real_p)

            df_processed["Predicted_MPG"] = final_mpg
            df_processed["Efficiency_Rating"] = df_processed["Predicted_MPG"].apply(lambda x: classify_efficiency(x)[0])
            
            log_to_ledger(df_processed["Predicted_MPG"].mean(), len(df_processed))
            st.success("Analysis Complete.")
            st.dataframe(df_processed)

else:
    st.header("Corporate Audit History")
    conn = sqlite3.connect("fleet_intelligence.db")
    st.table(pd.read_sql_query("SELECT timestamp, fleet_avg_mpg, total_assets FROM audit_ledger ORDER BY timestamp DESC", conn))
    conn.close()
