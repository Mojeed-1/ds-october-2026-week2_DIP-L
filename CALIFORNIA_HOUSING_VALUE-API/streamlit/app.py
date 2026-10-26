import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("HOUSING_API_URL", "http://127.0.0.1:8000").rstrip("/")
OCEAN_PROXIMITIES = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
]

st.set_page_config(
    page_title="California House Value",
    page_icon="🏠",
    layout="centered",
)

st.title("California House Value Predictor")
st.write(
    "Enter details about a California district to estimate its median house value."
)
st.caption(f"Prediction API: {API_BASE_URL}")

with st.form("housing_prediction_form"):
    st.subheader("Location")
    location_col1, location_col2 = st.columns(2)
    with location_col1:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-122.23,
            step=0.01,
        )
    with location_col2:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=37.88,
            step=0.01,
        )
    ocean_proximity = st.selectbox("Ocean proximity", OCEAN_PROXIMITIES)

    st.subheader("District details")
    age_col, income_col = st.columns(2)
    with age_col:
        housing_median_age = st.number_input(
            "Median house age (years)",
            min_value=0.0,
            value=41.0,
            step=1.0,
        )
    with income_col:
        median_income = st.number_input(
            "Median income (in tens of thousands of USD)",
            min_value=0.0001,
            value=8.3252,
            step=0.1,
            format="%.4f",
        )

    rooms_col, bedrooms_col = st.columns(2)
    with rooms_col:
        total_rooms = st.number_input(
            "Total rooms",
            min_value=0.0,
            value=880.0,
            step=1.0,
        )
    with bedrooms_col:
        total_bedrooms = st.number_input(
            "Total bedrooms",
            min_value=0.0,
            value=129.0,
            step=1.0,
        )

    population_col, households_col = st.columns(2)
    with population_col:
        population = st.number_input(
            "Population",
            min_value=0.0,
            value=322.0,
            step=1.0,
        )
    with households_col:
        households = st.number_input(
            "Households",
            min_value=0.0,
            value=126.0,
            step=1.0,
        )

    submitted = st.form_submit_button(
        "Predict house value",
        type="primary",
        use_container_width=True,
    )

if submitted:
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure FastAPI is running and HOUSING_API_URL is correct."
        )
        st.caption(str(exc))
    else:
        if response.ok:
            try:
                prediction = response.json()["predicted_median_house_value"]
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"The API returned an unexpected response: {exc}")
            else:
                st.success("Estimated median house value")
                st.metric("Predicted value", f"${prediction:,.2f}")
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except ValueError:
                error_detail = response.text
            st.error(f"Prediction failed ({response.status_code}): {error_detail}")
