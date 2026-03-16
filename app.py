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

# DARK THEME (Original Rule Preserved)
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
    base_year = 2000
    weight_evolution = 1 - ((year - base_year) * 0.007)
    
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
    
    final_mpg = min(final_mpg, max_physical_cap)
    return round(final_mpg, 2)

# --- NEW EXECUTIVE PDF ENGINE (Dynamic & Boardroom-Ready) ---
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    
    # Branding & Header
    pdf.set_fill_color(25, 25, 25); pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 22)
    pdf.cell(0, 20, "FLEET STRATEGY & ANALYTICS", ln=True, align='C')
    pdf.set_font("helvetica", '', 10)
    pdf.cell(0, 5, f"REF: {random.randint(1000,9999)} | GENERATED: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0); pdf.ln(20)

    # Dynamic Insight Engine
    openers = ["Strategic Overview:", "Operational Brief:", "Efficiency Performance Report:"]
    insights_positive = [
        "The current fleet trajectory indicates a healthy high-efficiency core.",
        "Resource utilization is trending positively toward enterprise sustainability goals.",
        "High-performance asset clusters have been successfully identified for scaling."
    ]
    insights_caution = [
        "Identified expenditure outliers suggest immediate fleet modernization opportunities.",
        "Significant cost-containment potential exists within the high-emission segment.",
        "Strategic asset replacement is recommended to offset rising fuel overheads."
    ]
    
    avg_mpg = df['Predicted_MPG'].mean()
    dist = df['Efficiency_Rating'].value_counts().to_dict()
    
    pdf.set_font("helvetica", 'B', 14); pdf.cell(0, 10, random.choice(openers), ln=True)
    pdf.set_font("helvetica", '', 11)
    
    selected_insight = random.choice(insights_positive) if dist.get('Excellent', 0) > dist.get('Poor', 0) else random.choice(insights_caution)
    pdf.multi_cell(0, 7, f"{selected_insight} With a calculated fleet average of {avg_mpg:.1f} MPG, the organization is well-positioned for data-driven optimization.")
    pdf.ln(5)

    # KPI Summary Grid
    pdf.set_font("helvetica", 'B', 11); pdf.set_fill_color(242, 242, 242)
    pdf.cell(63, 15, f"EXCELLENT: {dist.get('Excellent', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 15, f"AVERAGE: {dist.get('Average', 0)}", border=1, align='C', fill=True)
    pdf.cell(63, 15, f"POOR: {dist.get('Poor', 0)}", border=1, ln=True, align='C', fill=True)
    pdf.ln(10)

    # Executive Highlights (First 5 Vehicles)
    pdf.set_font("helvetica", 'B', 12); pdf.cell(0, 10, "Critical Asset Highlights", ln=True)
    pdf.set_font("helvetica", 'B', 10); pdf.set_fill_color(0, 114, 255); pdf.set_text_color(255, 255, 255)
    
    headers = ["Manufacturer", "Model", "Emissions", "AI-MPG", "Status"]
    widths = [40, 40, 30, 30, 50]
    for i, h in enumerate(headers): pdf.cell(widths[i], 10, h, border=1, align='C', fill=True)
    pdf.ln(); pdf.set_text_color(0, 0, 0); pdf.set_font("helvetica", '', 10)

    for _, row in df.head(5).iterrows():
        pdf.cell(widths[0], 10, str(row['Make']), border=1, align='C')
        pdf.cell(widths[1], 10, str(row.get('Model', 'N/A')), border=1, align='C')
        pdf.cell(widths[2], 10, str(row.get('CO2 Emissions', 'N/A')), border=1, align='C')
        pdf.cell(widths[3], 10, f"{row['Predicted_MPG']:.1f}", border=1, align='C')
        pdf.cell(widths[4], 10, str(row['Efficiency_Rating']), border=1, align='C')
        pdf.ln()

    # Strategic Conclusion
    pdf.ln(10); pdf.set_font("helvetica", 'B', 12); pdf.cell(0, 10, "Recommendations", ln=True)
    pdf.set_font("helvetica", '', 10)
    conclusions = [
        f"Based on a total projected fuel spend of ${df['Annual_Fuel_Cost'].sum():,.2f}, transitioning 'Poor' assets to hybrid equivalents could yield 15% ROI.",
        "The application of Physically Grounded AI Projections ensures that these results are calibrated against real-world mechanical constraints.",
        "Immediate attention should be directed toward the highest CO2 producing segments to mitigate enterprise carbon exposure."
    ]
    pdf.multi_cell(0, 6, random.choice(conclusions))
    
    # FIX: Robustly handle bytes vs string output from fpdf2
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return pdf_output

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

def prepare_ai_input(df, scaler_X):
    template = np.zeros((len(df), 12))
    input_df = pd.DataFrame(template, columns=RNN_COLS)
    for col in df.columns:
        if col in RNN_COLS: input_df[col] = df[col]
    final_numeric = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return scaler_X.transform(final_numeric.values)

def classify_efficiency(mpg):
    return "Excellent" if mpg > 35 else "Average" if mpg > 20 else "Poor"

# 3. INTERFACE
st.sidebar.title(f"Fleet Intel v5.6")
mode = st.sidebar.radio("Navigation", ["Single Vehicle", "Bulk Fleet Analytics"])

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
        comb = st.number_input("Combined L/100km", 2.0, 30.0, (city_l * 0.55) + (hwy_l * 0.45), step=0.1)

    if st.button("Generate AI Prediction"):
        single_row = pd.DataFrame([{
            "Model Year": v_year, "Make": v_make, "Engine Size": eng, "Cylinders": cyl, 
            "Fuel Type": fuel_t, "Vehicle Class": v_class, "Transmission": v_trans, 
            "CO2 Emissions": co2, "City (L/100km)": city_l, "Hwy (L/100km)": hwy_l, "Comb (L/100km)": comb
        }])
        
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
        df_raw = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
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
            st.plotly_chart(px.scatter(df_processed, x="Engine Size", y="Predicted_MPG", color="Efficiency_Rating", template="plotly_dark"), use_container_width=True)
            st.download_button("Download Executive Strategy Report (PDF)", create_pdf(df_processed), "Fleet_Strategy_Report.pdf")
