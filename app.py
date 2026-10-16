import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="USA House Price Predictor (Demo)",
    page_icon="🏠"
)

st.title("🏠 USA House Price Predictor")
st.write("Enter property details to estimate the price in USD.")

# ==========================================
# 2. LOAD DATA FROM EXCEL
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_excel("usa_housing_data.xlsx")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "location", "size", "bedrooms", "bathrooms",
        "property_type", "age", "price"
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing Excel columns: {', '.join(sorted(missing))}"
        )

    # Keep only the columns needed by the model
    df = df[required_columns].copy()

    # Convert numeric columns; invalid entries become missing
    numeric_columns = [
        "size", "bedrooms", "bathrooms", "age", "price"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing or invalid values
    df = df.dropna(subset=required_columns)

    # Remove impossible values
    df = df[
        (df["size"] > 0) &
        (df["bedrooms"] >= 1) &
        (df["bathrooms"] >= 1) &
        (df["age"] >= 0) &
        (df["price"] > 0)
    ].copy()

    # Clean categorical values
    df["location"] = df["location"].astype(str).str.strip()
    df["property_type"] = (
        df["property_type"].astype(str).str.strip()
    )

    df = df[
        (df["location"] != "") &
        (df["property_type"] != "")
    ]

    # Ensure whole-number bedroom counts
    df = df[
        (df["bedrooms"] % 1 == 0) &
        (df["bathrooms"] % 1 == 0)
    ].copy()

    df["bedrooms"] = df["bedrooms"].astype(int)
    df["bathrooms"] = df["bathrooms"].astype(int)

    if df.empty:
        raise ValueError(
            "No valid rows remain. Check your Excel data."
        )

    return df


try:
    data = load_data()
except Exception as e:
    st.error(f"Could not load the Excel file: {e}")
    st.stop()

st.success(f"Loaded {len(data):,} valid property records from Excel.")

with st.expander("Preview Excel data"):
    st.dataframe(data.head(10), use_container_width=True)

# ==========================================
# 3. PREPARE THE MODEL
# ==========================================

X = data.drop(columns=["price"])
y = data["price"]

numeric_features = [
    "size", "bedrooms", "bathrooms", "age"
]

categorical_features = [
    "location", "property_type"
]

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

# ==========================================
# 4. USER INPUT FORM
# ==========================================

st.subheader("Enter Property Details")

locations = sorted(data["location"].unique().tolist())
property_types = sorted(
    data["property_type"].unique().tolist()
)

with st.form("house_form"):

    location = st.selectbox(
        "US City / Location",
        locations
    )

    size = st.number_input(
        "House size (square feet)",
        min_value=200,
        max_value=15000,
        value=1500,
        step=100
    )

    bedrooms = st.selectbox(
        "Bedrooms",
        list(range(1, 8)),
        index=2
    )

    bathrooms = st.selectbox(
        "Bathrooms",
        list(range(1, 7)),
        index=1
    )

    property_type = st.selectbox(
        "Property type",
        property_types
    )

    age = st.number_input(
        "Property age (years)",
        min_value=0,
        max_value=150,
        value=10,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict House Price"
    )

# ==========================================
# 5. PREDICT PRICE
# ==========================================

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
        "This is a model estimate, not a professional valuation. "
        "Its accuracy depends on the quality and quantity "
        "of the Excel training data."
    )
