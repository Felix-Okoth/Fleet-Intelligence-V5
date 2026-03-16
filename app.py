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

# 1. SETUP & SECURITY INITIALIZATION
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

# DARK THEME
car_bg_url = "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=1920"
st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("{car_bg_url}"); background-size: cover; background-attachment: fixed; color: white; }}
[data-testid="stSidebar"] {{ background-color: rgba(0,0,0,0.85) !important; }}
label, p, .stMarkdown, h1, h2, h3, .stMetric, .stSelectbox {{ color: white !important; }}
div.stButton > button {{ background: linear-gradient(to right, #00c6ff, #0072ff); color: white; border-radius: 10px; font-weight: bold; width: 100%; }}
</style>
""", unsafe_allow_html=True)

# 2. RESOURCES & STOICHIOMETRIC LOGIC
RNN_COLS = ["Model Year", "Make", "Model", "Vehicle Class", "Engine Size", "Cylinders", "Transmission", "Fuel Type", "City (L/100km)", "Hwy (L/100km)", "Comb (L/100km)", "CO2 Emissions"]

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("fuel_efficiency_rnn_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_resources()

def classify_efficiency(mpg):
    if mpg > 35: return "Excellent", "rgba(0, 255, 0, 0.4)" 
    elif mpg > 20: return "Average", "rgba(255, 165, 0, 0.4)" 
    else: return "Poor", "rgba(255, 0, 0, 0.4)" 

def apply_hybrid_reality_logic(rnn_mpg, year, make, v_class, fuel_t, engine_size, cylinders, co2, source="Bulk"):
    fuel_chem = {"Regular": 8887, "Premium": 8887, "Diesel": 10180, "Ethanol": 5903}
    energy_constant = fuel_chem.get(fuel_t, 8887)
    chemical_truth_mpg = energy_constant / (max(co2, 1) * 1.609)
    friction_loss = (engine_size * 0.12) + (cylinders * 0.06)
    max_physical_cap = (68.0 / (1 + friction_loss))
    percent_variance = abs(rnn_mpg - chemical_truth_mpg) / chemical_truth_mpg
    log_to_performance_vault(rnn_mpg, chemical_truth_mpg, percent_variance, source)
    if percent_variance > 0.12:
        final_mpg = chemical_truth_mpg
    else:
        final_mpg = (rnn_mpg * 0.15) + (chemical_truth_mpg * 0.85)
    return round(min(final_mpg, max_physical_cap), 2)

# --- RESTORED ESG PDF ENGINE ---
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(25, 25, 25); pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 22)
    pdf.cell(0, 20, "FLEET STRATEGY & ESG ANALYTICS", ln=True, align='C')
    pdf.set_font("helvetica", '', 10); pdf.cell(0, 5, f"REF: {random.randint(1000,9999)} | GENERATED: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0); pdf.ln(15)
    pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, "Strategic Overview:", ln=True)
    pdf.set_font("helvetica", '', 11); avg_mpg = df['Predicted_MPG'].mean()
    pdf.multi_cell(0, 7, f"The fleet trajectory indicates a healthy high-efficiency core with an average of {avg_mpg:.1f} MPG."); pdf.ln(10)
    
    # Table Header
    pdf.set_font("helvetica", 'B', 10); pdf.set_fill_color(200, 200, 200)
    pdf.cell(40, 10, "Make", 1, 0, 'C', True); pdf.cell(30, 10, "Year", 1, 0, 'C', True)
    pdf.cell(40, 10, "MPG", 1, 0, 'C', True); pdf.cell(40, 10, "Rating", 1, 1, 'C', True)
    
    pdf.set_font("helvetica", '', 10)
    for i, row in df.head(15).iterrows():
        pdf.cell(40, 10, str(row.get('Make', 'N/A')), 1)
        pdf.cell(30, 10, str(row.get('Model Year', 'N/A')), 1)
        pdf.cell(40, 10, f"{row['Predicted_MPG']:.2f}", 1)
        pdf.cell(40, 10, str(row.get('Efficiency_Rating', 'N/A')), 1, 1)
        
    return pdf.output(dest='S').encode('latin-1')

def nlp_translator(df):
    df.columns = [sanitize_input(c.title().replace('_', ' ').strip()) for c in df.columns]
    mapping = {"Type Of Fuel": "Fuel Type", "Fueltype": "Fuel Type", "Emissions": "CO2 Emissions"}
    return df.rename(columns=mapping)

def prepare_ai_input(df, scaler_X):
    template = np.zeros((len(df), 12))
    input_df = pd.DataFrame(template, columns=RNN_COLS)
    for col in df.columns:
        if col in RNN_COLS: input_df[col] = df[col]
    return scaler_X.transform(input_df.apply(pd.to_numeric, errors='coerce').fillna(0).values)

# 3. INTERFACE
st.sidebar.title(f"Fleet Intel v6.5")
mode = st.sidebar.radio("Navigation", ["Single Vehicle", "Bulk Fleet Analytics"])

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
        
        rating, color = classify_efficiency(display_mpg)
        st.write(f"**{v_year} Efficiency Score**")
        st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 5px; width: 60%; margin: 10px 0; border-left: 5px solid white;">
                <h3 style="color: white; margin: 0; font-family: sans-serif;">{display_mpg:.2f} MPG - {rating}</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # RESTORED DOWNLOAD
        cleaned_df["Predicted_MPG"] = display_mpg
        cleaned_df["Efficiency_Rating"] = rating
        pdf_data = create_pdf(cleaned_df)
        st.download_button("Download ESG Report", data=pdf_data, file_name=f"ESG_Report_{v_make}.pdf", mime="application/pdf")

elif mode == "Bulk Fleet Analytics":
    st.header("Enterprise Analytics Engine")
    file = st.file_uploader("Upload Fleet Data", type=["csv", "xlsx"])
    if file:
        df_raw = pd.read_csv(file) if file.name.lower().endswith('.csv') else pd.read_excel(file)
        if st.button("Process Intelligence"):
            df_processed = nlp_translator(df_raw.copy())
            ai_in_raw = prepare_ai_input(df_processed, scaler_X)
            rnn_in = np.repeat(ai_in_raw[:, np.newaxis, :], 5, axis=1)
            raw_preds = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in))).flatten()
            
            final_mpg = [apply_hybrid_reality_logic(p, 2024, "Unknown", "Mid-Size", "Regular", 2.0, 4, 200) for p in raw_preds]
            df_processed["Predicted_MPG"] = final_mpg
            df_processed["Efficiency_Rating"] = df_processed["Predicted_MPG"].apply(lambda x: classify_efficiency(x)[0])
            
            log_to_ledger(df_processed["Predicted_MPG"].mean(), len(df_processed))
            st.dataframe(df_processed)
            
            pdf_data = create_pdf(df_processed)
            st.download_button("Download Bulk ESG Audit", data=pdf_data, file_name="Bulk_ESG_Audit.pdf", mime="application/pdf")
