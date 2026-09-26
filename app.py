
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("Tourism Package Purchase Prediction")
st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to purchase the tourism package."
)


# --------------------------------------------------
# Load Preprocessor and Model
# --------------------------------------------------

preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")


# --------------------------------------------------
# Customer Details
# --------------------------------------------------

st.header("Customer Information")

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

PreferredPropertyStar = st.number_input(
    "Preferred Property Star",
    min_value=1,
    max_value=5,
    value=3
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Unmarried"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0.0,
    max_value=50.0,
    value=3.0
)

Passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)

PitchSatisfactionScore = st.number_input(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

NumberOfFollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=20,
    value=3
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Purchase", type="primary"):

    input_data = pd.DataFrame({
        "Age": [Age],
        "TypeofContact": [TypeofContact],
        "CityTier": [CityTier],
        "Occupation": [Occupation],
        "Gender": [Gender],
        "NumberOfPersonVisiting": [NumberOfPersonVisiting],
        "PreferredPropertyStar": [PreferredPropertyStar],
        "MaritalStatus": [MaritalStatus],
        "NumberOfTrips": [NumberOfTrips],
        "Passport": [Passport],
        "OwnCar": [OwnCar],
        "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
        "Designation": [Designation],
        "MonthlyIncome": [MonthlyIncome],
        "PitchSatisfactionScore": [PitchSatisfactionScore],
        "ProductPitched": [ProductPitched],
        "NumberOfFollowups": [NumberOfFollowups],
        "DurationOfPitch": [DurationOfPitch]
    })

    # Apply the SAME preprocessing used during model training
    input_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_processed)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.success(
            "The customer is likely to purchase the tourism package."
        )
    else:
        st.warning(
            "The customer is unlikely to purchase the tourism package."
        )

