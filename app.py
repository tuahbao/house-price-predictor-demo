import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="USA House Price Predictor",
    page_icon="🏠"
)

st.title("🏠 USA House Price Predictor")
st.write("Enter property details to estimate the price in USD.")

# Sample data: illustrative values, not actual market data
data = pd.DataFrame({
    "location": [
        "New York", "New York", "New York",
        "Los Angeles", "Los Angeles", "Los Angeles",
        "Houston", "Houston", "Houston",
        "Chicago", "Chicago", "Chicago",
        "Miami", "Miami", "Miami",
        "Dallas", "Dallas", "Dallas",
        "New York", "Los Angeles"
    ],
    "size": [
        800, 1200, 1800, 900, 1500, 2200,
        1200, 1800, 2500, 1000, 1600, 2200,
        900, 1400, 2000, 1200, 1800, 2600,
        2500, 1100
    ],
    "bedrooms": [
        1, 2, 3, 2, 3, 4, 3, 3, 4, 2,
        3, 4, 2, 3, 4, 3, 4, 4, 4, 2
    ],
    "bathrooms": [
        1, 2, 2, 1, 2, 3, 2, 2, 3, 1,
        2, 3, 2, 2, 3, 2, 3, 3, 3, 1
    ],
    "property_type": [
        "Condo", "Condo", "Single-family",
        "Condo", "Single-family", "Single-family",
        "Single-family", "Single-family", "Single-family",
        "Condo", "Single-family", "Single-family",
        "Condo", "Single-family", "Single-family",
        "Single-family", "Single-family", "Single-family",
        "Single-family", "Condo"
    ],
    "age": [
        10, 20, 15, 8, 12, 5, 15, 10, 8, 20,
        12, 10, 8, 15, 5, 12, 8, 5, 10, 10
    ],
    "price": [
        500000, 750000, 1200000, 550000, 900000,
        1400000, 250000, 350000, 450000, 280000,
        400000, 550000, 450000, 650000, 1000000,
        300000, 450000, 650000, 1600000, 700000
    ]
})

# Verify that every column has the same number of rows
assert data.apply(len).nunique() == 1, "Columns have unequal lengths"

X = data.drop(columns=["price"])
y = data["price"]

preprocessor = ColumnTransformer([
    ("categories",
     OneHotEncoder(handle_unknown="ignore"),
     ["location", "property_type"]),
    ("numbers", "passthrough",
     ["size", "bedrooms", "bathrooms", "age"])
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X, y)

st.success("Model trained successfully!")

with st.form("house_form"):
    location = st.selectbox(
        "US City",
        ["New York", "Los Angeles", "Houston",
         "Chicago", "Miami", "Dallas"]
    )

    size = st.number_input(
        "House size (square feet)",
        min_value=200,
        max_value=15000,
        value=1500,
        step=100
    )

    bedrooms = st.selectbox(
        "Bedrooms", [1, 2, 3, 4, 5, 6, 7], index=2
    )

    bathrooms = st.selectbox(
        "Bathrooms", [1, 2, 3, 4, 5, 6], index=1
    )

    property_type = st.selectbox(
        "Property type", ["Condo", "Single-family"]
    )

    age = st.number_input(
        "Property age (years)",
        min_value=0,
        max_value=150,
        value=10
    )

    submitted = st.form_submit_button("Predict House Price")

if submitted:
    house = pd.DataFrame([{
        "location": location,
        "size": size,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_type": property_type,
        "age": age
    }])

    predicted_price = model.predict(house)[0]

    st.success(f"Estimated House Price: ${predicted_price:,.0f}")

    st.write("### Your assumptions")
    st.write(f"Location: {location}, USA")
    st.write(f"Size: {size:,} sq ft")
    st.write(f"Bedrooms: {bedrooms}")
    st.write(f"Bathrooms: {bathrooms}")
    st.write(f"Property type: {property_type}")
    st.write(f"Age: {age} years")

    st.warning(
        "This is an illustrative estimate based on only 20 "
        "sample records, not verified US property sales."
