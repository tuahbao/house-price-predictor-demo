
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="USA House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 USA House Price Predictor")
st.write(
    "Estimate a property's price using machine learning, "
    "compare it with similar properties, and understand "
    "the factors influencing the prediction."
)


# ============================================================
# 2. LOAD AND CLEAN DATA FROM EXCEL
# ============================================================

@st.cache_data
def load_data():

    # Read the Excel file.
    # Keep usa_housing_data.xlsx in the same folder as app.py.
    df = pd.read_excel("usa_housing_data.xlsx")

    # Standardize column names:
    # " Location " becomes "location".
    df.columns = df.columns.str.strip().str.lower()

    # Define the columns required by the application.
    required_columns = [
        "location",
        "size",
        "bedrooms",
        "bathrooms",
        "property_type",
        "age",
        "price"
    ]

    # Stop loading if any required column is missing.
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            "Missing Excel columns: "
            + ", ".join(sorted(missing))
        )

    # Keep only the columns used by the model.
    df = df[required_columns].copy()

    # Clean categorical columns before checking for missing values.
    for col in ["location", "property_type"]:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

        # Treat empty strings as missing values.
        df[col] = df[col].replace("", pd.NA)

    # Convert numeric columns to numeric values.
    # Invalid values such as "unknown" become NaN.
    numeric_columns = [
        "size",
        "bedrooms",
        "bathrooms",
        "age",
        "price"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove rows with missing required information.
    df = df.dropna(subset=required_columns)

    # Remove impossible or invalid property values.
    df = df[
        (df["size"] > 0)
        & (df["bedrooms"] >= 1)
        & (df["bathrooms"] >= 1)
        & (df["age"] >= 0)
        & (df["price"] > 0)
    ].copy()

    # Keep whole-number bedroom and bathroom counts.
    df = df[
        (df["bedrooms"] % 1 == 0)
        & (df["bathrooms"] % 1 == 0)
    ].copy()

    # Convert bedroom and bathroom counts to integers.
    df["bedrooms"] = df["bedrooms"].astype(int)
    df["bathrooms"] = df["bathrooms"].astype(int)

    # Reject a dataset with no valid records.
    if df.empty:
        raise ValueError(
            "No valid property records remain. "
            "Please check the Excel data."
        )

    return df


# Load the data and display a useful error if loading fails.
try:
    data = load_data()

except Exception as e:
    st.error(f"Could not load the Excel file: {e}")
    st.stop()


# Display the number of valid records.
st.success(
    f"Successfully loaded {len(data):,} valid property records."
)

# Allow the user to preview the cleaned dataset.
with st.expander("Preview Excel Dataset"):
    st.dataframe(
        data.head(20),
        use_container_width=True
    )


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

# X contains the property characteristics used for prediction.
# y contains the actual house prices that the model learns.
X = data.drop(columns=["price"])
y = data["price"]


# Define numerical features.
numeric_features = [
    "size",
    "bedrooms",
    "bathrooms",
    "age"
]

# Define categorical features.
categorical_features = [
    "location",
    "property_type"
]


# OneHotEncoder converts categories such as cities and property
# types into numerical columns that the model can understand.
#
# handle_unknown="ignore" allows the pipeline to handle an
# unfamiliar category without crashing during transformation.
preprocessor = ColumnTransformer(
    transformers=[
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
    ]
)


# Build the complete machine-learning pipeline.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 4. EVALUATE MODEL PERFORMANCE
# ============================================================

# Split the dataset into training and testing sets.
# The model will be evaluated on records it did not train on.
#
# This random split is suitable as a basic demonstration.
# If the dataset contains repeated listings or time-dependent
# prices, use a grouped or time-based split when appropriate.

if len(data) < 5:
    st.error(
        "At least five valid property records are needed "
        "for the train/test evaluation. Add more records "
        "to your Excel dataset."
    )
    st.stop()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train the model using only the training data.
model.fit(X_train, y_train)


# Predict the prices of the held-out test records.
test_predictions = model.predict(X_test)


# MAE: average absolute difference between predicted and
# actual prices, measured in USD.
mae = mean_absolute_error(
    y_test,
    test_predictions
)


# R²: measures predictive performance relative to a
# constant-mean baseline. It can be negative.
r2 = r2_score(
    y_test,
    test_predictions
)


# Retrain the final model on all valid records.
# This allows the prediction app to use all available data.
model.fit(X, y)


# ============================================================
# 5. CALCULATE OVERALL FEATURE IMPORTANCE
# ============================================================

# Random Forest provides an importance score for each
# transformed feature, including individual one-hot columns.
regressor = model.named_steps["regressor"]

transformed_feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)


# Create a table containing the transformed feature names
# and their importance scores.
importance_df = pd.DataFrame(
    {
        "feature": transformed_feature_names,
        "importance": regressor.feature_importances_
    }
)


# Group one-hot encoded columns back into their original
# feature categories.
def get_original_feature(feature_name):

    # Remove the ColumnTransformer prefix.
    name = feature_name.split("__", 1)[-1]

    if name.startswith("location_"):
        return "Location"

    elif name.startswith("property_type_"):
        return "Property type"

    elif name == "size":
        return "House size"

    elif name == "bedrooms":
        return "Bedrooms"

    elif name == "bathrooms":
        return "Bathrooms"

    elif name == "age":
        return "Property age"

    return name


importance_df["feature"] = (
    importance_df["feature"].apply(get_original_feature)
)


# Sum the importance of all encoded columns belonging to
# the same original feature.
importance_df = (
    importance_df
    .groupby("feature", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)


# Express importance as a percentage of total importance.
total_importance = importance_df["importance"].sum()

if total_importance > 0:
    importance_df["importance_pct"] = (
        importance_df["importance"]
        / total_importance
        * 100
    )
else:
    importance_df["importance_pct"] = 0.0


# ============================================================
# 6. DISPLAY MODEL PERFORMANCE
# ============================================================

st.divider()
st.header("📊 Model Performance")

st.write(
    "These metrics evaluate how the model performed on "
    "the held-out test dataset."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Mean Absolute Error (MAE)",
        f"${mae:,.0f}"
    )

with col2:
    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

with col3:
    st.metric(
        "Test Records",
        f"{len(X_test):,}"
    )


# Explain the metrics in plain English.
st.write(
    f"On the test dataset, the model's average absolute "
    f"prediction error was ${mae:,.0f}."
)

if r2 >= 0.75:
    st.write(
        "The R² score indicates that the model explains a "
        "relatively large share of the variation in the "
        "held-out house prices."
    )

elif r2 >= 0.40:
    st.write(
        "The model shows a moderate level of predictive "
        "performance on the held-out dataset."
    )

elif r2 >= 0:
    st.write(
        "The model has limited predictive performance "
        "relative to a constant-mean baseline. More data "
        "or improved features may help."
    )

else:
    st.write(
        "The model performed worse than a constant-mean "
        "baseline on the test dataset. Review the data "
        "quality and consider improving the model."
    )

st.caption(
    "These are test-set metrics, not a guarantee of future "
    "accuracy. The interpretation also depends on the "
    "dataset and how representative the test records are."
)


# ============================================================
# 7. USER INPUT FORM
# ============================================================

st.divider()
st.header("🏡 Enter Property Details")

# Use the available categories from the cleaned dataset.
locations = sorted(
    data["location"].unique().tolist()
)

property_types = sorted(
    data["property_type"].unique().tolist()
)


# Display a form so all inputs are submitted together.
with st.form("house_form"):

    # Location selection.
    location = st.selectbox(
        "US City / Location",
        locations
    )

    # House size.
    size = st.number_input(
        "House Size (square feet)",
        min_value=200,
        max_value=15000,
        value=1500,
        step=100
    )

    # Number of bedrooms.
    bedrooms = st.selectbox(
        "Bedrooms",
        list(range(1, 8)),
        index=2
    )

    # Number of bathrooms.
    bathrooms = st.selectbox(
        "Bathrooms",
        list(range(1, 7)),
        index=1
    )

    # Property type.
    property_type = st.selectbox(
        "Property Type",
        property_types
    )

    # Property age.
    age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=150,
        value=10,
        step=1
    )

    # Submit button.
    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 8. PREDICT HOUSE PRICE
# ============================================================

if submitted:

    # Build a one-row DataFrame using the same column names
    # and feature types used during model training.
    house = pd.DataFrame(
        [
            {
                "location": location,
                "size": size,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "property_type": property_type,
                "age": age
            }
        ]
    )


    # Obtain the Random Forest price estimate.
    predicted_price = float(model.predict(house)[0])


    st.divider()
    st.header("💰 House Price Analysis")


    # --------------------------------------------------------
    # 8A. Display predicted price
    # --------------------------------------------------------

    st.metric(
        label="Estimated House Price",
        value=f"${predicted_price:,.0f}"
    )

    st.caption(
        "This is a model estimate, not a guaranteed selling price."
    )


    # --------------------------------------------------------
    # 8B. Display property summary
    # --------------------------------------------------------

    st.subheader("1. Property Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "House Size",
        f"{size:,} sq ft"
    )

    col2.metric(
        "Bedrooms",
        f"{bedrooms}"
    )

    col3.metric(
        "Bathrooms",
        f"{bathrooms}"
    )

    st.write(f"**Location:** {location}")
    st.write(f"**Property Type:** {property_type}")
    st.write(f"**Property Age:** {age} years")


    # --------------------------------------------------------
    # 8C. Find comparable properties
    # --------------------------------------------------------

    st.subheader("2. Comparison With Similar Properties")

    # First select records from the same location and
    # property type.
    comparable = data[
        (data["location"] == location)
        & (data["property_type"] == property_type)
    ].copy()


    # Narrow the selection to properties within 20% of the
    # entered size and within one bedroom of the input.
    similar = comparable[
        (
            comparable["size"].between(
                size * 0.8,
                size * 1.2
            )
        )
        & (
            (comparable["bedrooms"] - bedrooms).abs() <= 1
        )
    ].copy()


    # If fewer than five properties are found, widen the
    # size range to 30% in each direction.
    comparison_note = ""

    if len(similar) < 5:

        similar = comparable[
            (
                comparable["size"].between(
                    size * 0.7,
                    size * 1.3
                )
            )
            & (
                (comparable["bedrooms"] - bedrooms).abs() <= 1
            )
        ].copy()

        comparison_note = (
            "The size range was widened to find more records."
        )


    # Require at least three comparable records before
    # presenting summary statistics.
    if len(similar) >= 3:

        # Median is less sensitive to unusually high or low
        # prices than the arithmetic mean.
        median_price = float(
            similar["price"].median()
        )

        # Calculate the 25th and 75th percentiles.
        q1 = float(
            similar["price"].quantile(0.25)
        )

        q3 = float(
            similar["price"].quantile(0.75)
        )


        # Display comparable-property statistics.
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Comparable Median",
            f"${median_price:,.0f}"
        )

        col2.metric(
            "25th Percentile",
            f"${q1:,.0f}"
        )

        col3.metric(
            "75th Percentile",
            f"${q3:,.0f}"
        )


        # Calculate the difference between the model estimate
        # and the median observed comparable-property price.
        difference = predicted_price - median_price

        if median_price > 0:
            difference_pct = (
                difference / median_price * 100
            )
        else:
            difference_pct = 0.0


        # Generate an explanation of the comparison.
        if difference > 0:

            comparison_text = (
                f"The model estimate is ${difference:,.0f} "
                f"({difference_pct:.1f}%) above the "
                "comparable-property median."
            )

        elif difference < 0:

            comparison_text = (
                f"The model estimate is ${abs(difference):,.0f} "
                f"({abs(difference_pct):.1f}%) below the "
                "comparable-property median."
            )

        else:

            comparison_text = (
                "The model estimate matches the "
                "comparable-property median."
            )


        st.write(comparison_text)

        st.write(
            f"This comparison uses {len(similar):,} records "
            "from the Excel dataset."
        )

        if comparison_note:
            st.caption(comparison_note)


        # Display the observed prices alongside the prediction.
        # The predicted price is a single model estimate;
        # it is not an actual transaction price.
        comparison_chart = pd.DataFrame(
            {
                "Price": (
                    similar["price"].tolist()
                    + [predicted_price]
                ),
                "Record Type": (
                    ["Comparable property"] * len(similar)
                    + ["Model estimate"]
                )
            }
        )

        st.write(
            "**Observed prices versus the model estimate**"
        )

        st.bar_chart(
            comparison_chart.groupby("Record Type")["Price"]
            .agg(["min", "median", "max"])
        )


        # Allow the user to inspect the underlying records.
        with st.expander(
            "View comparable property records"
        ):

            st.dataframe(
                similar[
                    [
                        "location",
                        "size",
                        "bedrooms",
                        "bathrooms",
                        "property_type",
                        "age",
                        "price"
                    ]
                ].sort_values("size"),
                use_container_width=True
            )


    else:

        # Do not invent comparison statistics if there are
        # too few similar properties.
        st.info(
            "There are not enough similar properties in "
            "the dataset to provide a meaningful comparison. "
            "The model prediction is still available."
        )


    # --------------------------------------------------------
    # 8D. Explain overall feature importance
    # --------------------------------------------------------

    st.subheader(
        "3. What Influences the Model's Predictions?"
    )

    # Display the overall importance of each original feature.
    feature_chart = (
        importance_df
        .set_index("feature")["importance_pct"]
    )

    st.bar_chart(feature_chart)

    # Display the exact percentages in a table.
    importance_display = importance_df[
        ["feature", "importance_pct"]
    ].copy()

    importance_display.columns = [
        "Feature",
        "Importance (%)"
    ]

    importance_display["Importance (%)"] = (
        importance_display["Importance (%)"].round(2)
    )

    st.dataframe(
        importance_display,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "These percentages describe overall feature importance "
        "across the Random Forest. They do not represent the "
        "percentage by which a feature raises or lowers this "
        "individual property's price."
    )


    # --------------------------------------------------------
    # 8E. Generate a plain-English explanation
    # --------------------------------------------------------

    st.subheader(
        "4. Explanation of the Estimated Price"
    )

    # Identify the three most important feature groups.
    top_features = (
        importance_df.head(3)["feature"].tolist()
    )

    top_features_text = ", ".join(top_features)

    # Explain how the prediction was obtained.
    st.write(
        f"The estimated price is ${predicted_price:,.0f}. "
        f"The Random Forest model learned patterns from "
        f"{len(data):,} property records in your Excel dataset "
        "and applied those patterns to the property details "
        "you entered."
    )

    st.write(
        f"The three most influential feature groups across "
        f"the overall model are {top_features_text}."
    )

    # Describe the local comparison when sufficient data exists.
    if len(similar) >= 3:

        st.write(
            f"The estimate is being compared with properties "
            f"of the same type in {location}. The comparison "
            f"contains {len(similar):,} records selected using "
            "similar size and bedroom criteria."
        )

        st.write(
            f"The comparable-property median is "
            f"${median_price:,.0f}, while the observed middle "
            f"50% of comparable prices falls between "
            f"${q1:,.0f} and ${q3:,.0f}."
        )

        st.write(
            "These statistics describe the observed dataset. "
            "They are not a confidence interval for the "
            "model's prediction."
        )

    else:

        st.write(
            "A reliable comparison could not be produced "
            "because too few similar records were available."
        )


    # Explain important limitations of the estimate.
    st.write(
        "The estimate may not fully account for renovations, "
        "interior condition, lot size, school districts, "
        "neighborhood differences, recent market changes, "
        "or other factors that are not included in the model."
    )


    # --------------------------------------------------------
    # 8F. Final disclaimer
    # --------------------------------------------------------

    st.warning(
        "This prediction is for demonstration and informational "
        "purposes only. The model's reliability depends on "
        "the quality, coverage, and representativeness of "
        "the Excel dataset. The comparable-property range is "
        "not a prediction interval or guarantee of market value."
    )
