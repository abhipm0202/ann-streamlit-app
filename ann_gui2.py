import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Colab ANN Trainer", layout="wide")

# Load Logos
nmis_logo = Image.open("nmis_logo.png")
d3m_logo = Image.open("d3mcolab_logo.png")

# Header with Logos and Title
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image(nmis_logo, use_container_width=True)
with col2:
    st.markdown("<h1 style='text-align: center;'>Colab ANN Trainer</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Welcome to ANN tool developed by D3MColab</h4>", unsafe_allow_html=True)
with col3:
    st.image(d3m_logo, use_container_width=True)

st.markdown("---")

# Sidebar for Inputs
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

    st.sidebar.header("2. Model Parameters")
    test_size = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.5, 0.2, 0.05)
    epochs = st.sidebar.number_input("Number of Epochs", min_value=10, max_value=1000, value=200, step=10)
    neuron_config = st.sidebar.text_input("Hidden Layers (comma separated)", value="10,10")
    neurons = [int(n) for n in neuron_config.split(",") if n.strip().isdigit()]

    st.sidebar.header("3. Training Configuration")
    solver = st.sidebar.selectbox("Select Training Algorithm (Solver)", ["adam", "sgd", "lbfgs"])
    loss_function = st.sidebar.selectbox("Loss Function (fixed in sklearn)", ["Mean Squared Error (MSE)"])
    st.sidebar.write("Note: `MLPRegressor` uses MSE internally for regression tasks.")

    if st.sidebar.button("Train Model"):
        try:
            X_vals = X.values
            Y_vals = Y.values.ravel() if Y.shape[1] == 1 else Y.values

            X_train, X_test, Y_train, Y_test = train_test_split(X_vals, Y_vals, test_size=test_size, random_state=42)

            model = MLPRegressor(hidden_layer_sizes=tuple(neurons), solver=solver,
                                 max_iter=epochs, random_state=42, verbose=False)
            model.fit(X_train, Y_train)

            Y_pred = model.predict(X_test)
            r2 = r2_score(Y_test, Y_pred)
            st.success(f"Model trained! R² Score: {r2:.3f}")

            # Plot loss curve and scatter
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].plot(model.loss_curve_, color='blue')
            axs[0].set_title("Training Loss Curve")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")

            axs[1].scatter(Y_test, Y_pred, alpha=0.6)
            axs[1].set_title("Actual vs Predicted")
            axs[1].set_xlabel("Actual")
            axs[1].set_ylabel("Predicted")

            st.pyplot(fig)

            st.session_state['model'] = model
        except Exception as e:
            st.error(f"Training failed: {e}")

    st.header("4. Make Predictions")

    # Manual Input Prediction
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

    # Excel File Prediction
    with st.expander("Predict from Excel File"):
        test_file = st.file_uploader("Upload Test X File", type=["xlsx"], key="test_x")
        if st.button("Predict from Excel"):
            try:
                model = st.session_state.get('model', None)
                if model:
                    test_X = pd.read_excel(test_file)
                    predictions = model.predict(test_X.values)
                    predictions = predictions.reshape(-1, 1) if len(predictions.shape) == 1 else predictions
                    result_df = pd.DataFrame(predictions, columns=[f"Y_Pred_{i+1}" for i in range(predictions.shape[1])])
                    st.write(result_df)

                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
                else:
                    st.error("Please train the model first.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
