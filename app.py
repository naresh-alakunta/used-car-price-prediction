
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# Load Saved Model and Preprocessor
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("/content/used_car_price_model_compressed.pkl")
    preprocessor = joblib.load("/content/used_car_preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

# -----------------------------
# Application Header
# -----------------------------
st.title("🚗 Used Car Price Prediction")
st.markdown(
    """
    Estimate the selling price of a used car using a
    **Machine Learning-powered valuation model**.

    Enter the vehicle details below to receive an estimated price.
    """
)

st.divider()

# -----------------------------
# Vehicle Input Form
# -----------------------------
with st.form("car_prediction_form"):

    st.subheader("🚘 Vehicle Details")

    left_col, right_col = st.columns(2)

    with left_col:
        brand = st.selectbox(
            "Car Brand",
            [
                "Maruti", "Hyundai", "Mahindra", "Tata", "Honda",
                "Ford", "Toyota", "Chevrolet", "Renault",
                "Volkswagen", "Nissan", "Skoda", "Datsun",
                "Mercedes-Benz", "BMW", "Fiat", "Audi", "Jeep",
                "Mitsubishi", "Volvo", "Jaguar", "Force", "Isuzu",
                "Ambassador", "Daewoo", "Land", "MG", "Kia",
                "Lexus", "Ashok", "Opel"
            ]
        )

        fuel = st.selectbox(
            "Fuel Type",
            ["Diesel", "Petrol", "CNG", "LPG"]
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            value=50000,
            step=1000
        )

        mileage = st.number_input(
            "Mileage (km/l or km/kg)",
            min_value=0.0,
            value=20.0,
            step=0.1
        )

        engine = st.number_input(
            "Engine Capacity (CC)",
            min_value=0.0,
            value=1197.0,
            step=10.0
        )

    with right_col:
        seller_type = st.selectbox(
            "Seller Type",
            ["Individual", "Dealer", "Trustmark Dealer"]
        )

        owner = st.selectbox(
            "Owner Type",
            [
                "First Owner",
                "Second Owner",
                "Third Owner",
                "Fourth & Above Owner",
                "Test Drive Car"
            ]
        )

        car_age = st.number_input(
            "Car Age (years)",
            min_value=0,
            value=5,
            step=1
        )

        max_power = st.number_input(
            "Maximum Power (bhp)",
            min_value=0.0,
            value=82.0,
            step=1.0
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            value=113.0,
            step=1.0
        )

        seats = st.number_input(
            "Number of Seats",
            min_value=1.0,
            value=5.0,
            step=1.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "💰 Predict Selling Price",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    input_data = pd.DataFrame({
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "torque": [torque],
        "seats": [seats],
        "car_age": [car_age],
        "brand": [brand]
    })

    try:
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]

        st.divider()
        st.subheader("📊 Price Estimation")

        st.success(
            f"Estimated Selling Price: ₹{prediction:,.0f}"
        )

        st.caption(
            "This is a machine learning estimate, not a guaranteed "
            "market price. Actual selling prices may vary."
        )

    except Exception as e:
        st.error(
            "Unable to generate the prediction. "
            "Please verify the vehicle details and try again."
        )
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Machine Learning Project | Python • Pandas • Scikit-learn • Streamlit"
)
