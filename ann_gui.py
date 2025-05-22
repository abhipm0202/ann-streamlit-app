import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, BinaryCrossentropy

st.set_page_config(layout="wide")
st.title("🧠 ANN Trainer and Predictor")

# Session state to persist model
if 'model' not in st.session_state:
    st.session_state.model = None

# --- Sidebar: Load Data ---
st.sidebar.header("📂 Load Data")

X_file = st.sidebar.file_uploader("Upload X (Features)", type=["xlsx"])
Y_file = st.sidebar.file_uploader("Upload Y (Targets)", type=["xlsx"])

if X_file and Y_file:
    X = pd.read_excel(X_file)
    Y = pd.read_excel(Y_file)

    st.subheader("✅ Uploaded Data Preview")
    st.write("Features (X):", X.head())
    st.write("Targets (Y):", Y.head())
else:
    st.warning("Please upload both X and Y Excel files to continue.")
    st.stop()

# --- Sidebar: Model Settings ---
st.sidebar.header("⚙️ Model Settings")

split_ratio = st.sidebar.slider("Train-Test Split", 0.1, 0.9, 0.2)
epochs = st.sidebar.number_input("Epochs", value=100)
neurons = st.sidebar.text_input("Neurons per Hidden Layer (comma-separated)", "10,10")
optimizer_choice = st.sidebar.selectbox("Optimizer", ["Adam", "SGD"])
loss_choice = st.sidebar.selectbox("Loss Function", ["MSE", "MAE", "BinaryCrossentropy"])

# --- Train Button ---
if st.sidebar.button("🚀 Train ANN"):
    try:
        X_vals = X.values
        Y_vals = Y.values

        X_train, X_test, Y_train, Y_test = train_test_split(X_vals, Y_vals, test_size=split_ratio, random_state=42)

        model = Sequential()
        for i, units in enumerate(map(int, neurons.split(','))):
            if i == 0:
                model.add(Dense(units, activation='relu', input_dim=X.shape[1]))
            else:
                model.add(Dense(units, activation='relu'))
        model.add(Dense(Y.shape[1] if len(Y.shape) > 1 else 1))

        optimizer = Adam() if optimizer_choice == "Adam" else SGD()
        loss_fn = {
            "MSE": MeanSquaredError(),
            "MAE": MeanAbsoluteError(),
            "BinaryCrossentropy": BinaryCrossentropy()
        }[loss_choice]
        model.compile(optimizer=optimizer, loss=loss_fn)

        history = model.fit(X_train, Y_train, epochs=epochs, verbose=0)

        Y_pred = model.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)

        st.session_state.model = model

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['loss'])
        ax[0].set_title("Training Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")

        ax[1].scatter(Y_test, Y_pred)
        ax[1].set_title(f"Predicted vs Actual (R² = {r2:.3f})")
        ax[1].set_xlabel("Actual")
        ax[1].set_ylabel("Predicted")

        st.pyplot(fig)
        st.success(f"✅ Model Trained. R² Score: {r2:.3f}")

    except Exception as e:
        st.error(f"Training failed: {e}")

# --- Predict Section ---
st.subheader("🔮 Make Predictions")

# Manual input
input_str = st.text_input("Enter comma-separated input values for X:")
if st.button("Predict Y"):
    try:
        input_values = np.array(list(map(float, input_str.split(',')))).reshape(1, -1)
        prediction = st.session_state.model.predict(input_values)
        st.success(f"Prediction: {prediction.flatten()}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# File input
test_file = st.file_uploader("Or upload an Excel file for X_test predictions", type=["xlsx"])
if test_file and st.button("Predict from File and Download"):
    try:
        test_X = pd.read_excel(test_file)
        preds = st.session_state.model.predict(test_X.values)
        pred_df = pd.DataFrame(preds, columns=[f"Y_Pred_{i+1}" for i in range(preds.shape[1])] if preds.ndim > 1 else ["Y_Pred"])
        st.download_button("Download Predictions as Excel", pred_df.to_excel(index=False), file_name="predictions.xlsx")
    except Exception as e:
        st.error(f"Prediction from file failed: {e}")
