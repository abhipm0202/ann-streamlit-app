
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Colab ANN Trainer", layout="wide")

# Load logos
nmis_logo = Image.open("nmis_logo.png")
d3m_logo = Image.open("d3mcolab_logo.png")

# Layout for logos and title
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image(nmis_logo, use_container_width=True)
with col2:
    st.markdown("<h1 style='text-align: center;'>Colab ANN Trainer</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Welcome to ANN GUI developed by D3MColab</h4>", unsafe_allow_html=True)
with col3:
    st.image(d3m_logo, use_container_width=True)

st.markdown("---")

# Sidebar for data upload
st.sidebar.header("1. Load Data")
uploaded_x = st.sidebar.file_uploader("Upload X (Features) Excel File", type=["xlsx"])
uploaded_y = st.sidebar.file_uploader("Upload Y (Target) Excel File", type=["xlsx"])

if uploaded_x and uploaded_y:
    X = pd.read_excel(uploaded_x)
    Y = pd.read_excel(uploaded_y)

    st.write("### Preview of X (features)")
    st.dataframe(X.head())
    st.write("### Preview of Y (target)")
    st.dataframe(Y.head())

    # Model configuration
    st.sidebar.header("2. Model Parameters")
    test_size = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.5, 0.2, 0.05)
    epochs = st.sidebar.number_input("Number of Epochs", min_value=10, max_value=1000, value=200, step=10)
    neuron_config = st.sidebar.text_input("Hidden Layers (comma separated)", value="10,10")
    neurons = [int(n) for n in neuron_config.split(",") if n.strip().isdigit()]

    st.sidebar.header("3. Training Configuration")
    solver = st.sidebar.selectbox("Select Training Algorithm (Solver)", ["adam", "sgd", "lbfgs"])
    st.sidebar.write("Note: `MLPRegressor` uses MSE internally for regression tasks.")

    if st.sidebar.button("Train Model"):
        try:
            X_vals = X.values
            Y_vals = Y.values if Y.shape[1] > 1 else Y.values.ravel()
            X_train, X_test, Y_train, Y_test = train_test_split(X_vals, Y_vals, test_size=test_size, random_state=42)

            model = MLPRegressor(hidden_layer_sizes=tuple(neurons), solver=solver, max_iter=epochs, random_state=42)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            st.session_state['model'] = model
            st.session_state['Y_test'] = Y_test
            st.session_state['Y_pred'] = Y_pred

            r2_scores = []
            if len(Y_pred.shape) == 1:
                r2_scores.append(r2_score(Y_test, Y_pred))
            else:
                for i in range(Y_pred.shape[1]):
                    r2_scores.append(r2_score(Y_test[:, i], Y_pred[:, i]))

            st.success("Model trained successfully.")
            for i, r2 in enumerate(r2_scores):
                st.markdown(f"**R² Score for Output {i+1}**: {r2:.4f}")

            selected_output = 0
            if len(r2_scores) > 1:
                selected_output = st.selectbox("Select Output to Visualize:", list(range(Y_pred.shape[1])))

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))

            axs[0].plot(model.loss_curve_)
            axs[0].set_title("Training Loss Curve")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")

            y_true = Y_test if len(Y_pred.shape) == 1 else Y_test[:, selected_output]
            y_pred = Y_pred if len(Y_pred.shape) == 1 else Y_pred[:, selected_output]

            axs[1].scatter(y_true, y_pred, alpha=0.6, label="Predictions")
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            axs[1].plot(y_true, p(y_true), "r--", label="Best Fit")
            axs[1].set_title("Actual vs Predicted")
            axs[1].set_xlabel("Actual")
            axs[1].set_ylabel("Predicted")
            axs[1].legend()

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Training failed: {e}")

    st.header("4. Make Predictions")

    with st.expander("Predict from Manual Input"):
        manual_input = st.text_input("Enter comma-separated values:")
        if st.button("Predict from Input"):
            try:
                model = st.session_state.get('model', None)
                if model:
                    input_array = np.array([float(x) for x in manual_input.split(",")]).reshape(1, -1)
                    pred = model.predict(input_array)
                    st.success(f"Predicted Y: {pred.flatten()}")
                else:
                    st.error("Please train the model first.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with st.expander("Predict from Excel File"):
        test_file = st.file_uploader("Upload Test X File", type=["xlsx"])
        if st.button("Predict from Excel") and test_file:
            try:
                model = st.session_state.get('model', None)
                if model:
                    test_X = pd.read_excel(test_file)
                    predictions = model.predict(test_X.values)
                    pred_df = pd.DataFrame(predictions, columns=[f"Y_Pred_{i+1}" for i in range(predictions.shape[1])] if predictions.ndim > 1 else ["Y_Pred"])
                    st.write(pred_df)
                    csv = pred_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
                else:
                    st.error("Please train the model first.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
