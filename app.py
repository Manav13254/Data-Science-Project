import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set wide layout
st.set_page_config(layout="wide")
st.title("🔬 Diabetes Prediction & EDA App")

# Load trained model and scaler
try:
    model = joblib.load("random_search_log_reg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.sidebar.success("✅ Model and Scaler Loaded!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model or scaler. Error: {e}")
    st.stop()

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        st.error("❌ 'diabetes.csv' file not found.")
        st.stop()

df = load_data()

# ----------- Sidebar Info -----------
st.sidebar.header("App Navigation")
tabs = st.sidebar.radio("Go to", ["EDA", "Predict"])

# ----------- EDA -----------
if tabs == "EDA":
    st.header("📊 Exploratory Data Analysis")

    if st.checkbox("Show raw data"):
        st.dataframe(df)

    if st.checkbox("Show data description"):
        st.dataframe(df.describe())

    if st.checkbox("Show correlation heatmap"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show feature distributions"):
        fig = df.hist(figsize=(12, 10))
        plt.tight_layout()
        st.pyplot(plt.gcf())

    if st.checkbox("Show bar plots of features by Outcome"):
        for col in df.columns:
            if col != "Outcome":
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x="Outcome", y=col, data=df, ax=ax)
                ax.set_title(f"Mean {col} by Outcome")
                st.pyplot(fig)

# ----------- Prediction Form -----------
elif tabs == "Predict":
    st.header("🧠 Predict Diabetes Based on User Input")

    with st.form("prediction_form"):
        st.subheader("🔢 Enter Feature Values:")

        glucose = st.number_input("Glucose", 0, 200)
        blood_pressure = st.number_input("Blood Pressure", 0, 150)
        skin_thickness = st.number_input("Skin Thickness", 0, 100)
        insulin = st.number_input("Insulin", 0, 900)
        bmi = st.number_input("BMI", 0.0, 70.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
        age = st.number_input("Age", 10, 120)

        submit = st.form_submit_button("Predict")

    if submit:
        # Corrected column name here
        input_df = pd.DataFrame([[glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                columns=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

        # Scaling input
        scaled_input = scaler.transform(input_df)
                
        # Making prediction
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][prediction]

        # Display result
        st.success(f"🩺 Prediction: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")
        st.info(f"🔢 Confidence: {prob:.2f}")
