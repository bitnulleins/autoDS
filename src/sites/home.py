import streamlit as st

st.header("👋 Welcome to autoDS!")

st.markdown(
"""
    autoDS stands for **Automatic Data Science** and is an Open-Source tool that automates as many phases of the data science life cycle as possible. This tool makes it possible to automatically train a regression model for a specific label, use it to make predictions and deploy it as an API. Upload the data and select a phase on the left. 👈

    ### Which steps are available?

    - **Upload data**: Load the pre-processed data set, select the label and prepare for next steps.
    - **EDA**: Explore data in deeply with Exploratory Data Analysis.
    - **Training**: A (new) best model is trained with AutoML.
    - **Prediction**: You can make predictions with an old or new model.
    - **Deployment**: You can deploy the trained model to an API, used for other systems.

    ### Overview


"""
)

st.image('./src/_static/images/pipeline.svg')