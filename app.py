import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# USA HOUSE PRICE PREDICTOR
# Illustrative sample data, not real sales
# ==========================================

st.set_page_config(
    page_title="USA House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 USA House Price Predictor")
st.write(
    "Enter your property assumptions to estimate "
    "a house price in US dollars (USD)."
)

# 1. Sample training data
data = pd.DataFrame({
    "location": [
        "New York", "New York", "New York",
        "Los Angeles", "Los Angeles", "Los Angeles",
        "Houston", "Houston", "Houston",
        "Chicago", "Chicago", "Chicago",
        "Miami", "Miami", "Miami",
        "Dallas", "Dallas", "Dallas",
        "New York", "Los Angeles", "Houston"
    ],
    "size": [
        800, 1200, 1800,
        900, 1500, 2200,
        1200, 1800, 2500,
        1000, 1600, 2200,
        900, 1400, 2000,
        1200, 1800, 2600,
        2500, 1100, 1500
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
        "Single-family", "Condo", "Single-family"
    ],
    "age": [
        10, 20, 15, 8, 12, 5, 15, 10, 8, 20,
        12, 10, 8, 15, 5, 12, 8, 5, 10, 10
    ],
    "price": [
        500000, 750000, 1200000,
        550000, 900000, 1400000,
        250000, 350000, 450000,
        280000, 400000, 550000,
        450000, 650000, 1000000,
        300000, 450000, 650000,
        1600000, 700000, 320000
    ]
})

# 2. Prepare model inputs
X = data.drop(columns="price")
y = data["price"]

numeric_features = [
    "size", "bedrooms", "bathrooms", "age"
]

categorical_features = [
    "location", "property_type"
]

# 3. Preprocess data
preprocessor = ColumnTransformer([
    (
        "categories",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "numbers",
        "passthrough",
        numeric_features
    )
])

# 4. Train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    )
])

model.fit(X, y)

# 5. User input form
st.subheader("Enter Property Details")

with st.form("house_form"):

    location = st.selectbox(
        "US City",
        [
            "New York",
            "Los Angeles",
            "Houston",
            "Chicago",
            "Miami",
            "Dallas"
        ]
    )

    size = st.number_input(
        "House size (square feet)",
        min_value=200,
        max_value=15000,
        value=1500,
        step=100
    )

    bedrooms = st.selectbox(
        "Number of bedrooms",
        [1, 2, 3, 4, 5, 6, 7]
    )

    bathrooms = st.selectbox(
        "Number of bathrooms",
        [1, 2, 3, 4, 5, 6]
    )

    property_type = st.selectbox(
        "Property type",
        ["Condo", "Single-family"]
    )

    age = st.number_input(
        "Property age (years)",
        min_value=0,
        max_value=150,
        value=10,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary"
    )

# 6. Make prediction
if submitted:

    new_house = pd.DataFrame([{
        "location": location,
        "size": size,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_type": property_type,
        "age": age
    }])

    predicted_price = model.predict(new_house)[0]

    st.success(
        f"Estimated House Price: ${predicted_price:,.0f}"
    )

    st.write("### Your assumptions")
    st.write(f"- Location: {location}, USA")
    st.write(f"- Size: {size:,} sq ft")
    st.write(f"- Bedrooms: {bedrooms}")
    st.write(f"- Bathrooms: {bathrooms}")
    st.write(f"- Property type: {property_type}")
    st.write(f"- Age: {age} years")

    st.warning(
        "Demo only: this estimate uses illustrative sample "
        "data, not verified US property sales. Do not use "
        "it for actual buying or selling decisions."
    )

st.divider()
st.caption(
    "Built with Python, Streamlit, and Scikit-learn."
)
