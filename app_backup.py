import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Price Predictor")
st.write("Predict house prices using Machine Learning")

# User Inputs
area = st.number_input("Area (sq ft)", value=5000)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
stories = st.slider("Stories", 1, 5, 2)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])

parking = st.slider("Parking Spaces", 0, 5, 1)

prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnished = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# Prepare input
input_data = pd.DataFrame({
    'area':[area],
    'bedrooms':[bedrooms],
    'bathrooms':[bathrooms],
    'stories':[stories],
    'mainroad':[1 if mainroad=="yes" else 0],
    'guestroom':[1 if guestroom=="yes" else 0],
    'basement':[1 if basement=="yes" else 0],
    'hotwaterheating':[1 if hotwaterheating=="yes" else 0],
    'airconditioning':[1 if airconditioning=="yes" else 0],
    'parking':[parking],
    'prefarea':[1 if prefarea=="yes" else 0],
    'furnishingstatus_semi-furnished':[1 if furnished=="semi-furnished" else 0],
    'furnishingstatus_unfurnished':[1 if furnished=="unfurnished" else 0]
})

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated House Price: ₹{prediction:,.0f}"
    )