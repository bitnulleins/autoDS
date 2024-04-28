from pathlib import Path

import os
import numpy as np
import pandas as pd
import streamlit as st
from pycaret.regression import *

from components.header import init
from dotenv import load_dotenv

init()
load_dotenv()

options = {}

if Path('./src/_static/model/deployed_model.pkl').exists():
    options['Deployed model'] = './src/_static/model/deployed_model'

if Path('./src/_static/model/uploaded_model.pkl').exists():
    options['Uploaded model'] = './src/_static/model/uploaded_model'

def select_model(option):
    regression = RegressionExperiment()
    return regression.load_model(options.get(option))

def render_form(df, target):
    def render_input(column, unique_items):
        sample = None if 'sample' not in st.session_state else st.session_state.sample.loc[column]

        #st.write(df.dtypes)

        if df[column].dtype == 'object':
            idx = np.where(unique_items==sample)[0]
            idx = None if len(idx) == 0 else int(idx[0])
            value = st.selectbox(column, unique_items, placeholder="Choose an option", index=idx)
        elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(df[column]):
            date = st.date_input(f"{column} date", value=sample, format="DD.MM.YYYY")
            time = st.time_input(f"{column} time", value=sample)
            value = pd.datetime.combine(date,time)
        else:
            if len(unique_items) == 2:
                value = st.toggle(column + "?", value=sample)
            else:
                value = st.number_input(
                    column,
                    value=sample,
                    min_value=unique_items.min(),
                    max_value=unique_items.max(),
                    help=f"Values from {unique_items.min()} to {unique_items.max()}")

        return value

    values = {}
    col1, col2 = st.columns(2)
    for idx, column in enumerate(np.sort(df.drop(columns=[target]).columns)):
        unique_items = df[column].sort_index().dropna().unique()

        if (idx % 2)+1 == 2:
            with col2: values[column] = render_input(column, unique_items)
        else:
            with col1: values[column] = render_input(column, unique_items)

    return pd.DataFrame([values])

st.header("🔎 Prediction Preview")

st.write("Compare deployed and other uploaded models.")

if "df" in st.session_state:
    df = st.session_state.df

    if "selected_model" in st.session_state:
        target = "FLUG.baggage"
        new_df = render_form(df, target)

        if len(new_df.dropna()) > 0:
            regression = st.session_state.reg
            model = regression.load_model('./src/_static/model/deployed_model')

            with st.sidebar:
                y_pred = predict_model(model, data=new_df)['prediction_label']
                st.metric("Predict " + target, value=y_pred.iloc[0].round().astype(int))
        else:
            empty_columns = new_df.isna().any(axis=0)
            empty_columns = empty_columns[empty_columns == True].index.to_list()
            st.error(f"There still missing values: {', '.join(empty_columns)}")
    else:
        st.error("No model selected.")

    st.divider()

    def load_sample(): st.session_state.sample = df.sample(1, random_state=None).iloc[0]

    col1, col2 = st.columns(2)

    with col1: st.session_state.selected_model = st.radio("Select model", options.keys())
    with col2: 
        st.markdown("Load sample data")
        st.button("Load", on_click=load_sample, use_container_width=True)

    file = st.file_uploader("Upload model (as pickle):", type="pkl")

    if file:
        with open('./src/_static/model/uploaded_model.pkl', 'wb') as tmp:
            tmp.write(file.getvalue())

            regression = RegressionExperiment()
            model = regression.load_model('./src/_static/model/uploaded_model')

            st.write(model)

else:
    st.warning("First upload a dataset and/or train a model.")