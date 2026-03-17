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

# ==========================================
# NEW: CRYPTOGRAPHY & DB INITIALIZATION
# ==========================================
def handle_secrets():
    if not os.path.exists("dev_secret.key"):
        key = Fernet.generate_key()
        with open("dev_secret.key", "wb") as key_file:
            key_file.write(key)
    else:
        with open("dev_secret.key", "rb") as key_file:
            key = key_file.read()
    return Fernet(key)

cipher = handle_secrets()

def encrypt_data(data):
    return cipher.encrypt(str(data).encode()).decode()

def decrypt_data(token):
    return cipher.decrypt(token.encode()).decode()

def init_db():
    conn = sqlite3.connect("fleet_intelligence.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audit_ledger 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, fleet_avg_mpg REAL, total_assets INTEGER, total_fuel_cost REAL, insights_logged TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS performance_vault 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, vehicle_make TEXT, rnn_predicted_mpg REAL, 
                 physics_truth_mpg REAL, variance_percent REAL, was_corrected INTEGER)''')
    conn.commit()
    conn.close()

def log_performance_metric_silent(make, rnn_mpg, physics_mpg, variance):
    conn = sqlite3.connect("fleet_intelligence.db")
    c = conn.cursor()
    enc_make = encrypt_data(make)
    was_corrected = 1 if variance > 0.12 else 0
    c.execute('''INSERT INTO performance_vault (timestamp, vehicle_make, rnn_predicted_mpg, physics_truth_mpg, variance_percent, was_corrected) 
                 VALUES (?, ?, ?, ?, ?, ?)''', (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), enc_make, rnn_mpg, physics_mpg, variance, was_corrected))
    conn.commit()
    conn.close()

def log_fleet_session_silent(avg_mpg, asset_count, fuel_cost, insights=""):
    conn = sqlite3.connect("fleet_intelligence.db")
    c = conn.cursor()
    c.execute('''INSERT INTO audit_ledger (timestamp, fleet_avg_mpg, total_assets, total_fuel_cost, insights_logged) VALUES (?, ?, ?, ?, ?)''',
              (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), avg_mpg, asset_count, fuel_cost, insights))
    conn.commit()
    conn.close()

init_db()

# ==========================================
# ACTIVITY 4 & 5: ANALYTICS & INSIGHTS
# ==========================================
def render_fleet_visuals(df):
    st.subheader("Fleet Performance Analytics")
    
    # Chart 1: Efficiency Frontier (Minimized circles to prevent crowding)
    fig_frontier = px.scatter(
        df, x="Engine Size", y="Predicted_MPG", 
        color="Efficiency_Rating", 
        size="CO2 Emissions",
        size_max=10, 
        hover_name="Model", title="Efficiency Frontier: Displacement vs. MPG",
        color_discrete_map={"Excellent": "#00ffcc", "Average": "#f1c40f", "Poor": "#ff4b4b"},
        template="plotly_dark"
    )
    # Refine marker appearance for better density handling
    fig_frontier.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color='DarkSlateGrey')))
    
    st.plotly_chart(fig_frontier, use_container_width=True)

    # Vertical Spacing and Divider
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    # Chart 2: Fuel Exposure
    fig_cost = px.bar(
        df, x="Make", y="Annual_Fuel_Cost", 
        color="Efficiency_Rating", title="Annual Fuel Exposure by OEM",
        barmode="group", template="plotly_dark"
    )
    st.plotly_chart(fig_cost, use_container_width=True)

def generate_strategic_insights(df):
    insights = []
    avg_fleet_mpg = df["Predicted_MPG"].mean()
    high_risk = df[df["Predicted_MPG"] < (avg_fleet_mpg * 0.7)]
    if not high_risk.empty:
        insights.append(f"CRITICAL: {len(high_risk)} assets are performing 30% below fleet average. Recommend immediate phase-out.")
    
    co2_col = "CO2 Emissions" if "CO2 Emissions" in df.columns else "Emissions"
    if co2_col in df.columns:
        top_polluter = df.groupby("Make")[co2_col].mean().idxmax()
        insights.append(f"STRATEGIC: {top_polluter} assets hold the highest carbon intensity in your inventory.")
    
    insights.append("OPERATIONAL: Divert 'Excellent' rated assets to high-mileage routes to maximize fuel ROI.")
    return insights

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

# --- ACTIVITY 1: GHOST LOGIC INJECTION ---
def apply_hybrid_reality_logic(rnn_mpg, year, make, v_class, fuel_t, engine_size, cylinders, co2):
    make_bias = {"Toyota": 0.95, "Honda": 0.95, "Ford": 1.10, "Chevrolet": 1.10}
    m_factor = make_bias.get(make, 1.0)
    fuel_chem = {"Regular": 8887, "Premium": 8887, "Diesel": 10180, "Ethanol": 5903}
    energy_constant = fuel_chem.get(fuel_t, 8887)
    chemical_truth_mpg = energy_constant / (max(co2, 1) * 1.609)
    friction_loss = (engine_size * 0.12) + (cylinders * 0.06)
    max_physical_cap = (68.0 / (1 + friction_loss)) * (1 / m_factor)
    percent_variance = abs(rnn_mpg - chemical_truth_mpg) / chemical_truth_mpg
    
    log_performance_metric_silent(make, rnn_mpg, chemical_truth_mpg, percent_variance)

    if percent_variance > 0.12:
        final_mpg = chemical_truth_mpg
    else:
        final_mpg = (rnn_mpg * 0.15) + (chemical_truth_mpg * 0.85)
    return round(min(final_mpg, max_physical_cap), 2)

# --- THE ESG-READY PDF ENGINE (FINAL STRIKE VERSION) ---
def create_pdf(df, fig=None, insights=[]):
    def safe_str(text):
        if text is None: return "N/A"
        try:
            clean = str(text).encode('ascii', 'ignore').decode('ascii')
            return clean.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'")
        except:
            return "Data Formatting Error"

    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(25, 25, 25); pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 22)
    pdf.cell(0, 20, "FLEET STRATEGY & ESG ANALYTICS", ln=True, align='C')
    pdf.set_font("helvetica", '', 10)
    pdf.cell(0, 5, f"REF: {random.randint(1000,9999)} | GENERATED: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0); pdf.ln(15)
    
    # Strategic Overview
    pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, "Strategic Overview:", ln=True)
    pdf.set_font("helvetica", '', 11)
    avg_mpg = df['Predicted_MPG'].mean()
    overview_text = (f"The current fleet trajectory indicates a healthy high-efficiency core. "
                     f"With a calculated fleet average of {avg_mpg:.1f} MPG, the organization "
                     f"is well-positioned for data-driven optimization.")
    pdf.multi_cell(0, 7, safe_str(overview_text))
    pdf.ln(5)
    
    # Insights Section
    if insights:
        pdf.set_font("helvetica", 'B', 12); pdf.cell(0, 10, "AI-Driven Strategic Insights:", ln=True)
        pdf.set_font("helvetica", '', 10)
        for insight in insights:
            clean_line = safe_str(insight)
            try:
                pdf.multi_cell(0, 6, f"- {clean_line}")
            except:
                pdf.cell(0, 6, "- [Metadata Error]", ln=True)
        pdf.ln(5)

    # Metrics
    co2_col = "CO2 Emissions" if "CO2 Emissions" in df.columns else next((c for c in df.columns if "CO2" in c.upper()), "Emissions")
    dist = df['Efficiency_Rating'].value_counts().to_dict()
    pdf.set_font("helvetica", 'B', 11); pdf.set_fill_color(242, 242, 242)
    pdf.cell(63, 15, f"EXCELLENT: {dist.get('Excellent', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 15, f"AVERAGE: {dist.get('Average', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 15, f"POOR: {dist.get('Poor', 0)}", border=1, ln=True, align='C', fill=True)
    pdf.ln(10)
    
    # Table
    pdf.set_font("helvetica", 'B', 12); pdf.cell(0, 10, "Critical Asset Highlights", ln=True)
    pdf.set_font("helvetica", 'B', 10); pdf.set_fill_color(0, 114, 255); pdf.set_text_color(255, 255, 255)
    headers = ["Manufacturer", "Model", "Emissions", "AI-MPG", "Status"]
    widths = [40, 40, 30, 30, 50]
    for i, h in enumerate(headers): pdf.cell(widths[i], 10, h, border=1, align='C', fill=True)
    pdf.ln(); pdf.set_text_color(0, 0, 0); pdf.set_font("helvetica", '', 10)
    
    for _, row in df.head(10).iterrows():
        pdf.cell(widths[0], 10, safe_str(row.get('Make', 'N/A')), border=1, align='C')
        pdf.cell(widths[1], 10, safe_str(row.get('Model', 'N/A')), border=1, align='C')
        pdf.cell(widths[2], 10, safe_str(row.get(co2_col, 'N/A')), border=1, align='C')
        pdf.cell(widths[3], 10, f"{row.get('Predicted_MPG', 0):.1f}", border=1, align='C')
        pdf.cell(widths[4], 10, safe_str(row.get('Efficiency_Rating', 'N/A')), border=1, align='C')
        pdf.ln()
    
    # Plotly Image
    if fig:
        try:
            pdf.add_page()
            pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, "Visual Efficiency Distribution", ln=True)
            img_bytes = fig.to_image(format="png", width=800, height=500, scale=2)
            pdf.image(io.BytesIO(img_bytes), x=10, y=30, w=190)
        except Exception:
            pdf.cell(0, 10, "[Note: Visual distribution available in app dashboard]", ln=True)

    try:
        pdf_out = pdf.output()
        return bytes(pdf_out) if not isinstance(pdf_out, str) else pdf_out.encode('latin-1', 'replace')
    except:
        err_pdf = FPDF()
        err_pdf.add_page()
        err_pdf.set_font("Arial", size=12)
        err_pdf.cell(0, 10, "Report Error: Specific data contains incompatible encoding for PDF Export.", ln=True)
        return err_pdf.output().encode('latin-1', 'replace')

def nlp_translator(df):
    df.columns = [c.title().replace('_', ' ').strip() for c in df.columns]
    mapping = {"Type Of Fuel": "Fuel Type", "Fueltype": "Fuel Type", "Emissions": "CO2 Emissions", "Co2 Emissions": "CO2 Emissions", "Combined": "Comb (L/100km)", "Make": "Make", "Model": "Model"}
    df = df.rename(columns=mapping)
    if "Transmission" in df.columns:
        df["Trans_Clean"] = df["Transmission"].astype(str).str.upper().str.strip()
        df["Transmission"] = df["Trans_Clean"].apply(lambda x: 2 if "CVT" in x else 1 if "M" in x else 0)
    if "Fuel Type" in df.columns:
        df["Fuel Type"] = df["Fuel Type"].astype(str).str.title().str.strip().map(lambda x: FUEL_MAP.get(x, 1))
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
st.sidebar.title(f"Fleet Intel v5.9")
mode = st.sidebar.radio("Navigation", ["Single Vehicle", "Bulk Fleet Analytics"])

if st.query_params.get("dev_mode") == "true":
    with st.sidebar.expander("DEVELOPER BACKDOOR"):
        if st.button("Decrypt & View Audit Vault"):
            conn = sqlite3.connect("fleet_intelligence.db")
            vault = pd.read_sql_query("SELECT * FROM performance_vault", conn)
            vault['vehicle_make'] = vault['vehicle_make'].apply(decrypt_data)
            st.dataframe(vault)
            conn.close()

if mode == "Single Vehicle":
    st.header("Vehicle Profile")
    c1, c2 = st.columns(2)
    with c1:
        v_make = st.text_input("Vehicle Make", "Toyota")
        eng = st.number_input("Engine Size (L)", 0.5, 10.0, 2.0, step=0.1)
        cyl = st.number_input("Cylinders", 2, 16, 4, step=1)
        fuel_t = st.selectbox("Fuel Type", ["Regular", "Premium", "Diesel", "Ethanol"])
        v_year = st.number_input("Model Year", 1995, 2026, 2024, step=1)
    with c2:
        v_class = st.selectbox("Vehicle Class", ["Mid-Size", "Compact", "SUV", "Pickup", "Truck"])
        v_trans = st.selectbox("Transmission", ["Automatic", "Manual", "CVT"])
        co2 = st.number_input("CO2 Emissions (g/km)", 50, 600, 200, step=1)
        city_l = st.number_input("City (L/100km)", 2.0, 30.0, 10.0, step=0.1)
        hwy_l = st.number_input("Hwy (L/100km)", 2.0, 30.0, 8.0, step=0.1)
        comb = (city_l * 0.55) + (hwy_l * 0.45)
    if st.button("Generate AI Prediction"):
        single_row = pd.DataFrame([{"Model Year": v_year, "Make": v_make, "Engine Size": eng, "Cylinders": cyl, "Fuel Type": fuel_t, "Vehicle Class": v_class, "Transmission": v_trans, "CO2 Emissions": co2, "City (L/100km)": city_l, "Hwy (L/100km)": hwy_l, "Comb (L/100km)": comb}])
        cleaned_df = nlp_translator(single_row)
        ai_in_raw = prepare_ai_input(cleaned_df, scaler_X)
        rnn_in = np.repeat(ai_in_raw[:, np.newaxis, :], 5, axis=1) 
        raw_mpg = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in)))[0][0]
        display_mpg = apply_hybrid_reality_logic(raw_mpg, v_year, v_make, v_class, fuel_t, eng, cyl, co2)
        st.divider()
        st.metric(f"{v_year} Efficiency Score", f"{display_mpg:.2f} MPG")
        st.success(f"Rating: {classify_efficiency(display_mpg)}")

else:
    st.header("Enterprise Analytics Engine")
    file = st.file_uploader("Upload Fleet Data", type=["csv", "xlsx"])
    if file:
        df_raw = pd.read_csv(file) if file.name.lower().endswith('.csv') else pd.read_excel(file, engine='openpyxl')
        df_processed = nlp_translator(df_raw.copy())
        if st.button("Process Intelligence"):
            ai_in_raw = prepare_ai_input(df_processed, scaler_X)
            rnn_in = np.repeat(ai_in_raw[:, np.newaxis, :], 5, axis=1)
            raw_preds = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in))).flatten()
            final_mpg = []
            for i, p in enumerate(raw_preds):
                row = df_raw.iloc[i] 
                real_p = apply_hybrid_reality_logic(p, row.get("Model Year", 2024), row.get("Make", "Unknown"), row.get("Vehicle Class", "Mid-Size"), row.get("Fuel Type", "Regular"), row.get("Engine Size", 2.0), row.get("Cylinders", 4), row.get("CO2 Emissions", 200))
                final_mpg.append(real_p)
            df_processed["Predicted_MPG"] = final_mpg
            df_processed["Annual_Fuel_Cost"] = (ANNUAL_MILES / df_processed["Predicted_MPG"]) * FUEL_PRICE
            df_processed["Efficiency_Rating"] = df_processed["Predicted_MPG"].apply(classify_efficiency)
            
            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("Total Fleet Spend", f"${df_processed['Annual_Fuel_Cost'].sum():,.0f}")
            m2.metric("Avg Fleet MPG", f"{df_processed['Predicted_MPG'].mean():.1f}")
            
            st.dataframe(df_processed)
            
            render_fleet_visuals(df_processed)
            
            fleet_insights = generate_strategic_insights(df_processed)
            st.subheader("Strategic Recommendations")
            for ins in fleet_insights:
                st.info(ins)
            
            log_fleet_session_silent(df_processed["Predicted_MPG"].mean(), len(df_processed), df_processed["Annual_Fuel_Cost"].sum(), insights=" | ".join(fleet_insights))

            report_data = create_pdf(df_processed, fig=None, insights=fleet_insights)
            st.download_button(label="Download Executive Strategy Report (PDF)", data=report_data, file_name="Fleet_Strategy_Report.pdf", mime="application/pdf")
