import os

import streamlit as st
from dotenv import load_dotenv
import datetime as dt
from zipfile import ZipFile
from pycaret.regression import *

load_dotenv()

from components.header import init

init()

st.title("⚙️ Deploy API")

if "reg" in st.session_state:
    with st.spinner("Init api..."):
        model = st.session_state.reg.load_model('./src/_static/model/deployed_model')
        name = st.session_state.target.lower().replace('.','_') + "_api"
        st.session_state.reg.create_api(
            model, f'./src/api/{name}',
            host=os.environ.get('API_HOST'),
            port=int(os.environ.get('API_PORT'))
        )

        # Fix "Timestamp" error: Timestamp -> pd.Timestamp

        # Read in the file
        with open(f'./src/api/{name}.py', 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('Timestamp(', 'pd.Timestamp(')

        # Write the file out again
        with open(f'./src/api/{name}.py', 'w') as file:
            file.write(filedata)

        st.session_state.reg.create_docker(f'./src/api/{name}')
    
        st.session_state.api = 'Deployed'
        st.success('Create API and Docker files. Finished API deployment!')

    st.markdown(f'''
    ### Step 1: Start API

    To start API, run the following command:

    ```shell
    python ./src/api/baggage_preview_api.py
    ```

    or in pipenv environment:
    
    ```shell
    python -m pipenv run api
    ```
                
    or build Docker image and run as container:

    ```shell
    docker image build -f "Dockerfile" -t {name}:latest .
    docker run -i -t {name}:latest
    ```
    ''')

    zipFile = f"{name}_{dt.datetime.today().strftime('%Y%m%d%H%M%S')}.zip"
    with ZipFile('deployment.zip', mode='w') as f:
        f.write('Dockerfile')
        f.write(f'./src/api/{name}.py')
        f.write(f'./src/api/{name}.pkl')

    with open("deployment.zip", "rb") as fp:
        st.download_button(
            "⬇️ Download deployment files",
            data=fp,
            mime='application/zip',
            file_name=zipFile
        )

    st.markdown(f'''
    The API will start at `{os.environ.get('API_HOST')}` in port {os.environ.get('API_PORT')}

    ### Step 2: Call ReST API

    Do POST request `curl -X "POST" 'http://{os.environ.get('API_HOST')}:{os.environ.get('API_PORT')}/predict'` with request body:

    ''')

    sample = st.session_state.df.sample(1)
    feature = sample[st.session_state.target].values[0]
    json = sample.drop(columns=st.session_state.target).to_dict('records')[0]
    st.json(json)

    st.markdown(f"The response returns the predicted value for **{st.session_state.target}**:")

    st.json({
        "prediction" : feature
    })

    st.divider()

    st.link_button("🗒️ Open documentation", "http://" + os.environ.get('API_HOST')+":"+os.environ.get('API_PORT')+"/docs")
else:
    st.warning("First train a new model and deploy it.")