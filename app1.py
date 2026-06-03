import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI-Powered Real Estate Market Intelligence Platform",
    page_icon="🏠",
    layout="wide"
)

# ----------------------------
# Load Dataset & Model
# ----------------------------
df = pd.read_csv("data/housing.csv")
model = joblib.load("models/model.pkl")

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset Explorer",
        "Visualizations",
        "Feature Importance",
        "Price Prediction",
        "About"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    st.title("🏠 AI-Powered Real Estate Market Intelligence Platform")

    st.markdown("""
    Welcome to the Real Estate Analytics Dashboard.

    This platform combines:
    - Data Analysis
    - Interactive Visualizations
    - Machine Learning
    - House Price Prediction
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Properties",
        df.shape[0]
    )

    col2.metric(
        "Features",
        df.shape[1]
    )

    col3.metric(
        "Average Price",
        f"₹{int(df['price'].mean()):,}"
    )

    st.divider()

    st.subheader("Project Highlights")

    st.write("""
    - Machine Learning Model: Random Forest Regressor
    - Dataset Size: 545 Properties
    - Features Used: 12
    - Dashboard Framework: Streamlit
    - Visualization Library: Plotly
    - Prediction Engine: Scikit-Learn
    """)

# =====================================================
# DATASET EXPLORER
# =====================================================
elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# =====================================================
# VISUALIZATIONS
# =====================================================
elif page == "Visualizations":

    st.title("📈 Real Estate Analytics")

    fig1 = px.histogram(
        df,
        x="price",
        nbins=30,
        title="House Price Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.scatter(
        df,
        x="area",
        y="price",
        size="bedrooms",
        hover_data=["bathrooms"],
        title="Area vs Price"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.box(
        df,
        x="bedrooms",
        y="price",
        title="Bedrooms vs Price"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    furnishing_avg = (
        df.groupby("furnishingstatus")["price"]
        .mean()
        .reset_index()
    )

    fig4 = px.bar(
        furnishing_avg,
        x="furnishingstatus",
        y="price",
        title="Average Price by Furnishing Status"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# =====================================================
# PRICE PREDICTION
# =====================================================
# =====================================================
# FEATURE IMPORTANCE
# =====================================================
# =====================================================
# FEATURE IMPORTANCE
# =====================================================
elif page == "Feature Importance":

    st.title("📊 Feature Importance Analysis")

    feature_names = [
        'area',
        'bedrooms',
        'bathrooms',
        'stories',
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'parking',
        'prefarea',
        'furnishingstatus_semi-furnished',
        'furnishingstatus_unfurnished'
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Most Important Factors Affecting House Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    ### Key Insights

    - Area is the most important factor affecting house prices.
    - Bathrooms strongly influence property value.
    - Air conditioning contributes to higher prices.
    - Parking and stories also impact valuation.
    """)

elif page == "Price Prediction":

    st.title("🤖 House Price Prediction")

    st.write(
        "Enter property details below to predict the estimated house price."
    )

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq ft)",
            min_value=500,
            value=5000
        )

        bedrooms = st.slider(
            "Bedrooms",
            1,
            10,
            3
        )

        bathrooms = st.slider(
            "Bathrooms",
            1,
            10,
            2
        )

        stories = st.slider(
            "Stories",
            1,
            5,
            2
        )

    with col2:

        parking = st.slider(
            "Parking Spaces",
            0,
            5,
            1
        )

        mainroad = st.selectbox(
            "Main Road Access",
            ["yes", "no"]
        )

        airconditioning = st.selectbox(
            "Air Conditioning",
            ["yes", "no"]
        )

        prefarea = st.selectbox(
            "Preferred Area",
            ["yes", "no"]
        )

    input_df = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [1 if mainroad == "yes" else 0],
        'guestroom': [0],
        'basement': [0],
        'hotwaterheating': [0],
        'airconditioning': [1 if airconditioning == "yes" else 0],
        'parking': [parking],
        'prefarea': [1 if prefarea == "yes" else 0],
        'furnishingstatus_semi-furnished': [0],
        'furnishingstatus_unfurnished': [0]
    })

    if st.button("Predict House Price"):

        prediction = model.predict(input_df)[0]

        st.success(
            f"Estimated Property Value: ₹{prediction:,.0f}"
        )

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":

    st.title("ℹ️ About This Project")

    st.write("""
    ### Real Estate Market Intelligence Platform

    This project demonstrates a complete Data Science workflow:

    1. Data Collection
    2. Data Cleaning
    3. Exploratory Data Analysis
    4. Machine Learning Model Training
    5. Model Deployment
    6. Interactive Dashboard Development

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - Plotly
    - Streamlit

    ### Model

    Random Forest Regressor was used for house price prediction.

    ### Author

    Data Science Project
    """)