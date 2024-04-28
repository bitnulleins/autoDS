import datetime as dt
import os

import streamlit as st
from dotenv import load_dotenv
from pycaret.regression import *
from components.header import init
from streamlit_shap import st_shap

init()
load_dotenv()

def experiment(df, target):
    regression = RegressionExperiment()
    regression.setup(
        session_id              = 42,
        data                    = df,
        log_experiment          = False,
        experiment_name         = "regression",
        target                  = target,
        # Feature Engineering
        create_date_columns     = ['year','month','week','weekday','hour'],
        # Ausreißer entfernen
        remove_outliers         = True,
        outliers_method         = 'iforest',
        # Missing values
        numeric_imputation      = 'mean',
        categorical_imputation  = 'mode',
        normalize               = True,
        normalize_method        = 'zscore',
        # One-Hot-Encoding
        max_encoding_ohe        = 1000,
        # Train-Test Split
        train_size              = 0.7,
        # Training
        fold                    = 10,
        fold_strategy           = 'kfold',
    )
    return regression

def generate_performance(regression, best_model_name):
    print("Generate plots")
    metrics = regression.pull().loc['Mean']
    residuals = regression.plot_model(best, plot = 'residuals', save='./src/_static/images/plots/')
    errors = regression.plot_model(best, plot = 'error', save='./src/_static/images/plots/')
    feature_all = regression.plot_model(best, plot = 'feature_all', save='./src/_static/images/plots/')
    learning = regression.plot_model(best, plot = 'learning', save='./src/_static/images/plots/')
    print("Generate XAI plot")
    if best_model_name in ['rf','et','lightgbm','dt','xgboost']:
        interpret = regression.interpret_model(best, save = './src/_static/images/plots/')
    else:
        interpret = regression.interpret_model(best, plot = 'pfi', save = './src/_static/images/plots/')

    return metrics, residuals, errors, feature_all, learning, interpret

st.title("🤖 Training with AutoML")

if 'df' in st.session_state and 'target' in st.session_state:

    if 'deployed' not in st.session_state:

        st.subheader("Step 1: Find best model")

        df = st.session_state.df.sample(st.session_state.sample_size)
        target = st.session_state.target

        metric = None
        metric_input = st.selectbox('What do you want to optimize?',
            ('Low error','High correlation'), index=None)

        if metric_input == 'Low error':
            metric = 'RMSE'
        elif metric_input == 'High correlation':
            metric = 'R2'

        if metric is not None:

            if st.session_state.target:
                with st.spinner('Prepare first training...'):
                    regression = experiment(df.sample(st.session_state.sample_size), target)

                with st.expander("Experimental overview"):
                    st.dataframe(regression.pull(), use_container_width=True, hide_index=True)

            st.write(f"Find best model with sample of {st.session_state.sample_size} rows.")
            
            with st.spinner("Compare models with sample..."):
                best = regression.compare_models(sort=metric)

            st.markdown("**Results:**")

            leaderboard = regression.pull()
            st.dataframe(leaderboard, use_container_width=True, hide_index=True)

            st.success(f"Best model: {best}")

            st.subheader("Step 2: Train best model")

            st.write("Train best model with full data.")

            with st.spinner("Train final model with all data..."):

                with st.expander("Experimental overview"):
                    st.dataframe(regression.pull(), use_container_width=True, hide_index=True)

                if len(leaderboard) > 1:
                    best_model_name = leaderboard[leaderboard['TT (Sec)'] < leaderboard['TT (Sec)'].mean()].iloc[0].name
                else:
                    best_model_name = leaderboard.iloc[0].name

                best = regression.create_model(best_model_name)

                st.success("Finished training best model!")

            st.subheader(f"Step 3: Tune best model. This may take a few minutes.")

            with st.spinner("Boosting model (optimization)..."):
                best = regression.tune_model(
                    best,
                    n_iter=50,
                    optimize = metric,
                    return_tuner= False,
                    choose_better = True
                )

            st.dataframe(regression.pull(), use_container_width=True)
            
            st.success(f"Tuned best model!")

            st.subheader("Step 4: Model performance")

            with st.spinner("Generate performance metrics..."):
                metrics, residuals, errors, feature_all, learning, interpret = generate_performance(regression, best_model_name)

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Metrics", "Residuals", "Erros", "Learning", "Feature Importance", "Interpret"])

            with tab1:
                index = 0
                col1, col2, col3 = st.columns(3)
                for name, value in zip(metrics.index, metrics.to_numpy()):
                    with globals()[f"col{(index%3)+1}"]:
                        st.metric(label=name, value=value)
                        index += 1

            with tab2:
                st.image(residuals)

            with tab3:
                st.image(errors)

            with tab4:
                st.image(learning)

            with tab5:
                st.image(feature_all)

            with tab6:
                if best_model_name in ['rf','et','lightgbm','dt','xgboost']:
                    st.write("SHAP-value")
                    st.image('./src/_static/images/plots/SHAP summary.png')
                else:
                    st.write("Permutation Feature Importance")
                    st.write(interpret)

            st.subheader("Step 5: Finalize model")

            with st.spinner('Train final model...'):
                best = regression.create_model(best_model_name)
                st.success(f"Finalized best model!")

            st.subheader("Step 6: Deploy model")

            def click_button():
                st.session_state.saved = True
                st.session_state.deployed = True
                st.session_state.reg = regression
                st.session_state.best = best
                st.session_state.best_name = best_model_name

            st.button("Deploy model", type="primary", on_click=click_button)

    else:

        if "saved" in st.session_state:
            st.balloons()
            st.session_state.reg.save_model(st.session_state.best, './src/_static/model/deployed_model')
            st.success("Successfully saved model.")

            # Clean up session_state
            del st.session_state.saved
            del st.session_state.best
        else:
            st.info("Model already saved.")

        st.write("You're done. Next step?")

        left_col, mid_col, right_col = st.columns(3)

        with left_col:
            st.page_link(
                page='pages/preview.py',
                icon = '🔎',
                label='Make predictions'
            )

        with mid_col:
            st.page_link(
                page='pages/api.py',
                icon = '⚙️',
                label='Deploy API',
            )

        with right_col:
            with open('./src/_static/model/deployed_model.pkl', "rb") as fp:
                st.download_button(
                    "⬇️ Download model",
                    data=fp,
                    file_name=f"model_{dt.datetime.today().strftime('%Y%m%d%H%M%S')}.pkl",
                    use_container_width=True
                )

else:
    st.warning("You've to upload and prepare a dataframe first.")