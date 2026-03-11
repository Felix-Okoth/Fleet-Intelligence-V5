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

# STEP 9: ENTERPRISE AUTHENTICATION
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

# STEP 8: DARK CAR THEME
car_bg_url = "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=1920"
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("{car_bg_url}");
    background-size: cover; background-attachment: fixed; color: white;
}}
[data-testid="stSidebar"] {{ background-color: rgba(0,0,0,0.85) !important; }}
label, p, .stMarkdown, h1, h2, h3, .stMetric {{ color: white !important; }}
div.stButton > button {{
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white; border-radius: 10px; font-weight: bold; width: 100%;
}}
</style>
""", unsafe_allow_html=True)

# 2. RESOURCES & LOGIC
# UPDATED FOR CLOUD DEPLOYMENT: Logs now save in the app's home directory
DRIVE_LOG_PATH = "user_logs.csv" 
FUEL_PRICE, ANNUAL_MILES, CO2_PER_GALLON = 4.50, 15000, 8.88
RNN_COLS = ["Model Year", "Make", "Model", "Vehicle Class", "Engine Size", "Cylinders",
            "Transmission", "Fuel Type", "City (L/100km)", "Hwy (L/100km)", "Comb (L/100km)", "CO2 Emissions"]

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("fuel_efficiency_rnn_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_resources()

def log_user_data(engine, cyl, mpg, rating):
    try:
        log_entry = pd.DataFrame({
            "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "engine_size": [engine], "cylinders": [cyl],
            "predicted_mpg": [round(mpg, 2)], "rating": [rating]
        })
        header = not os.path.isfile(DRIVE_LOG_PATH)
        log_entry.to_csv(DRIVE_LOG_PATH, mode='a', header=header, index=False)
    except: pass

def classify_efficiency(mpg):
    return "Excellent" if mpg > 35 else "Average" if mpg > 20 else "Poor"

def math_refinement(raw_mpg, row):
    return raw_mpg * 0.96 if "AWD" in str(row.get('Drive Type', '')).upper() else raw_mpg

def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(200, 10, "Fleet Intelligence Report", ln=True, align='C')
    pdf.set_font("Arial", size=12); pdf.ln(10)
    pdf.cell(200, 10, f"Avg Efficiency: {df['Predicted_MPG'].mean():.2f} MPG", ln=True)
    pdf.cell(200, 10, f"Total Annual Cost: ${df['Annual_Fuel_Cost'].sum():,.2f}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# 3. INTERFACE
st.sidebar.title("Fleet Intel v5.0")
mode = st.sidebar.radio("Navigation", ["Single Vehicle", "Bulk Fleet Analytics"])

if mode == "Single Vehicle":
    st.header("Quick Vehicle Profiler")
    c1, c2 = st.columns(2)
    with c1:
        eng = st.number_input("Engine Size (L)", 0.5, 10.0, 2.0)
        cyl = st.number_input("Cylinders", 2, 16, 4)
    with c2:
        co2 = st.number_input("CO2 Emissions (g/km)", 50, 600, 200)
        comb = st.number_input("Combined L/100km", 2.0, 30.0, 9.0)

    if st.button("Generate AI Prediction"):
        features = np.array([[2026, 0, 0, 0, eng, cyl, 0, 0, comb+1, comb-1, comb, co2]])
        scaled = scaler_X.transform(features)
        rnn_in = np.repeat(scaled[:, np.newaxis, :], 5, axis=1)
        raw_mpg = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in)))[0][0]
        final_mpg = math_refinement(raw_mpg, {'Engine Size': eng})
        rating = classify_efficiency(final_mpg)
        log_user_data(eng, cyl, final_mpg, rating)
        st.metric("Efficiency Score", f"{final_mpg:.2f} MPG")
        st.success(f"Rating: {rating} | Data Saved")

else:
    st.header("Enterprise Analytics Engine")
    file = st.file_uploader("Upload Fleet Data", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if st.button("Process Intelligence"):
            df.columns = [c.title().strip() for c in df.columns]
            ai_in = scaler_X.transform(df.reindex(columns=RNN_COLS, fill_value=0))
            rnn_in = np.repeat(ai_in[:, np.newaxis, :], 5, axis=1)
            raw_preds = np.expm1(scaler_y.inverse_transform(model.predict(rnn_in))).flatten()
            df["Predicted_MPG"] = [math_refinement(p, r) for p, r in zip(raw_preds, df.to_dict('records'))]
            df["Annual_Fuel_Cost"] = (ANNUAL_MILES / df["Predicted_MPG"]) * FUEL_PRICE
            df["Efficiency_Rating"] = df["Predicted_MPG"].apply(classify_efficiency)

            k1, k2 = st.columns(2)
            k1.metric("Total Fleet Spend", f"${df['Annual_Fuel_Cost'].sum():,.0f}")
            k2.metric("Avg Fleet MPG", f"{df['Predicted_MPG'].mean():.1f}")
            st.dataframe(df)

            fig = px.scatter(df, x="Engine Size", y="Predicted_MPG", color="Efficiency_Rating", template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download Executive PDF", create_pdf(df), "fleet_report.pdf")
