import pandas as pd
import os
import plotly.express as px
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer

from components.header import init
from dotenv import load_dotenv

load_dotenv()
init()


st.header("🗄️ Dataset")
@st.cache_data
def load_data():
    """Generate dataframe with sample size."""
    df = pd.read_parquet(file)
    return df

if 'df' in st.session_state:
    st.info("Dataset successfully uploaded!")

    st.write("Want to restart and load another dataset?")

    status = st.button("Reset", type="primary")

    if status:
        del st.session_state.target
        del st.session_state.df
        if 'eda' in st.session_state: del st.session_state.eda
        if 'reg' in st.session_state: del st.session_state.reg
        if 'api' in st.session_state: del st.session_state.api
        if 'deployed' in st.session_state: del st.session_state.deployed
        if 'sample_size' in st.session_state: del st.session_state.sample_size
        st.switch_page("./pages/upload.py")
        
else:
    st.subheader("Step 1: Upload")

    with st.spinner('Upload file...'):
        file = st.file_uploader("Prepared Dataset:", type="parquet")

    if file:
        if 'df' not in st.session_state:
            df = load_data()
            st.success("Upload complete.")

        st.markdown('**How many random samples do you want to use?**')

        number = st.number_input(
            f"Insert a sample size (0 - {len(df)})",
            value=len(df),
            min_value=0,
            max_value=len(df),
            step=1)
            
        st.session_state.sample_size = number
        percentage = round(st.session_state.sample_size / len(df) * 100)

        st.write("Amount entries: ", df.shape[0])
        st.write("Amount features: ", df.shape[1])
        st.write("Sample size in %: ", percentage)

        if st.session_state.sample_size > 0:

            st.subheader("Step 2: Preview dataset")

            st.write("Lets look at the first entry.")

            filtered_df = dataframe_explorer(df, case=False)
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            st.subheader("Step 3: Select data")

            st.write("Select label and columns for training.")

            columns = st.multiselect("Columns", df.columns.to_list(), default=df.columns.to_list())
            target = st.selectbox("Label", placeholder="Choose item...", options=df.columns, index=None)

            if st.button('Save'):
                if target:
                    st.info(f"Choose {st.session_state.sample_size} random samples ({percentage}%).")

                    st.session_state.target = target
                    st.session_state.df = df[columns]

                    st.markdown('**Distribution of target value**')

                    # Create distplot with custom bin_size
                    fig = px.histogram(df, x=target)

                    # Plot!
                    st.plotly_chart(fig, use_container_width=True)

                    st.success("Finished data aquisition. Go to next step.")

                    col1, col2 = st.columns(2)

                    with col2: st.page_link(page='pages/automl.py', icon='🤖')
                    with col1: st.page_link(page='pages/eda.py', icon='📊')
                else:
                    st.error("You've to choose a target value first!")