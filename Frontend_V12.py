import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne

# Backend URL
BACKEND_URL = "https://b7bf-103-161-223-11.ngrok-free.app"

st.set_page_config(page_title="Brain Health Prediction", layout="wide")

# Function to check backend status
def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status/")
        if response.status_code == 200:
            return response.json().get("status", "Unknown status"), "✅ Online"
        else:
            return "Backend not reachable", "❌ Offline"
    except requests.exceptions.RequestException:
        return "Backend not reachable", "❌ Offline"

# Function to upload and predict
def upload_and_predict(file):
    if file is not None:
        with st.spinner("Processing file..."):
            files = {"file": file.getvalue()}
            response = requests.post(f"{BACKEND_URL}/predict/", files=files)
            if response.status_code == 200:
                return response.json().get("classification", "Error processing file")
            else:
                return "Error: Unable to get prediction"
    return "No file uploaded"

# Function to plot brain topology
def plot_brain_topology(df):
    st.subheader("🧠 Brain Topology Map")
    
    if "Channel" not in df.columns or "Dominant Frequency (Hz)" not in df.columns:
        st.error("The uploaded file must contain 'Channel' and 'Dominant Frequency (Hz)' columns.")
        return
    
    df = df.dropna(subset=["Channel", "Dominant Frequency (Hz)"])
    df["Dominant Frequency (Hz)"] = pd.to_numeric(df["Dominant Frequency (Hz)"], errors='coerce') -17.5
    df = df.dropna()
    
    unique_channels = df["Channel"].unique()
    freq_values = df.groupby("Channel")["Dominant Frequency (Hz)"].mean()
    
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = [ch for ch in unique_channels if ch in montage.ch_names]
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types=["eeg"] * len(ch_names))
    info.set_montage(montage)
    
    topo_values = [freq_values.get(ch, 0) for ch in ch_names]
    fig, ax = plt.subplots(figsize=(5, 5))
    mne.viz.plot_topomap(topo_values, info, axes=ax, show=False, cmap="Reds", contours=0)
    st.pyplot(fig)

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        "<h1 style='text-align: center; font-size: 36px; font-weight: bold; margin-top: -30px;'>Navigation</h1>",
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
            height: 60px;
            font-size: 20px;
            font-weight: bold;
        }
        .result-text {
            font-size: 24px;
            font-weight: bold;
            color: #2E86C1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🏠 Home", key="home_btn"):
        st.session_state["selected_tab"] = "Home"
    if st.button("Check Backend Status", key="check_status_btn"):
        st.session_state["selected_tab"] = "Check Backend Status"
    if st.button("Upload & Predict", key="upload_predict_btn"):
        st.session_state["selected_tab"] = "Upload & Predict"

if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = "Home"

if st.session_state["selected_tab"] == "Home":
    st.title("🧠 Brain Health Prediction System")
    st.markdown("### Predict brain health status based on EEG data.")
    st.markdown("""  
        About This Project:

        - Objective: Study and apply brainwave activity to improve neurological condition understanding, treatment, and mental health management.  
        - EEG Overview: A non-invasive technique that measures brain electrical activity, providing insights into brain function.  
        - Scope:
        - Medical Diagnostics: EEG aids in diagnosing brain injuries and unstablilties.  
        - Treatment Enhancement: Helps refine treatment protocols for neurological and mental health conditions.  
         
          
    """)
    st.subheader("🏠 Welcome to the Brain Health Prediction System")
    st.write("Use the navigation panel to check backend status or upload data for prediction.")

elif st.session_state["selected_tab"] == "Check Backend Status":
    st.subheader("🔄 Check Backend Status")
    if st.button("Check Status"):
        status, emoji = check_backend_status()
        st.markdown(f"<p class='result-text'>Status: {emoji} {status}</p>", unsafe_allow_html=True)

elif st.session_state["selected_tab"] == "Upload & Predict":
    st.subheader("📂 Upload CSV for Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(df.head())

        if st.button("🧠 Show Brain Topology"):
            plot_brain_topology(df)

        if st.button("🔍 Predict"):
            result = upload_and_predict(uploaded_file)
            st.markdown(f"<p class='result-text'> Prediction Result: {result}</p>", unsafe_allow_html=True)
