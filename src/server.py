import pycaret
import os

import streamlit as st
import base64
import numpy as np
from st_pages import add_page_title, get_nav_from_toml

@st.cache_data
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def build_markup_for_logo(
    png_file,
    background_position="50% 0em",
    margin="4em",
    image_width="85%",
    image_height="",
    width="100%"
):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                [data-testid="stSidebarNav"] {
                    background-image: url("data:image/svg+xml;base64,%s");
                    background-repeat: no-repeat;
                    background-position: %s;
                    padding-top: %s;
                    background-size: %s %s;
                }
                [data-testid="stSidebarNavItems"] {
                    margin-top: 20px;
                }
                h1 {
                    position: fixed;
                    top: 60px;
                    background: #fff;
                    z-index: 999;
                    width: %s;
                }
            </style>
            """ % (
        binary_string,
        background_position,
        margin,
        image_width,
        image_height,
        width,
    )

def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )


# Page Title
st.set_page_config(
    page_title='autoDS',
    page_icon='./src/_static/images/favicon.png',
    menu_items = {
        'About':
        """
            **Version:**    1.0
            
            **Author:**     Finn Dohrn
        """
    }
)

# Custom CSS
st.markdown(
    """
<style>

    .stPageLink a {
        padding: 0.25rem 0.75em 0.25em 0.75em;
        border-radius: 0.5rem;
        min-height: 38.4px;
        margin-top: 6px;
        width:100%;
        line-height: 1.6;
        margin-top:6px;
        border: 1px solid rgba(51, 51, 51, 0.2);
    }
    iframe {
        border:1px solid rgba(51, 51, 51, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)

add_logo("src/_static/images/logo.svg")

with st.sidebar:
    progress_text = "Tasks completed"

    my_bar = st.progress(0, text=progress_text)

    if 'df' in st.session_state:
        my_bar.progress(0.25, text=progress_text)
    
    if 'eda' in st.session_state:
        my_bar.progress(0.5, text=progress_text)
    
    if 'reg' in st.session_state:
        my_bar.progress(0.75, text=progress_text)
    
    if 'api' in st.session_state:
        my_bar.progress(100, text=progress_text)

# Fixed Random seed
np.random.seed(int(os.environ.get("RANDOM_SEED")))

# Pages
nav = get_nav_from_toml(".streamlit/pages.toml")
pg = st.navigation(nav)
add_page_title(pg)
pg.run()