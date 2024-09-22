import pandas as pd
import os
import plotly.express as px
import streamlit as st

from ydata_profiling import ProfileReport
from dotenv import load_dotenv

@st.cache_data
def generate_profile():
    pr = ProfileReport(
        st.session_state.df.sample(st.session_state.sample_size),
        samples=None,
        correlations=None,
        missing_diagrams=None,
        duplicates=None,
        interactions=None,
    )
    pr.config.html.style.primary_colors = ['#DA1C30']
    return pr

st.title("📊 Exploratory Data Analysis (EDA)")

if 'df' in st.session_state:
    st.write(f"Explore target {st.session_state.target}")

    if "eda" not in st.session_state:
        with st.spinner("Creating profiling report..."):
            pr = generate_profile()
            pr.to_file("./src/_static/eda_report.html")
            st.session_state.eda = True

    html = open("./src/_static/eda_report.html", 'r', encoding='utf-8')
    st.components.v1.html(html.read(), height=600, scrolling=True)
    st.session_state.eda = True
else:
    st.warning("Upload dataset first for EDA.")