import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Tourism Package Prediction", layout="wide")

# --- Load the Model ---
# The model was saved as 'best_tourism_prediction_model_v1.joblib'
MODEL_PATH = "best_tourism_prediction_model_v1.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# --- App Header ---
st.title("🌐 Visit with Us: Wellness Package Predictor")
st.markdown("""
This app predicts whether a customer is likely to purchase the new **Wellness Tourism Package** 
based on their profile and interaction history.
""")

if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the model is trained and available in the deployment folder.")
else:
    # --- Input UI ---
    st.sidebar.header("Customer Attributes")
    
    # Features matching the training columns
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
    occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    person_visiting = st.sidebar.slider("Number of Persons Visiting", 1, 10, 3)
    followups = st.sidebar.slider("Number of Follow-ups", 1, 10, 3)
    product_pitched = st.sidebar.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
    property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    trips = st.sidebar.number_input("Number of Trips", 1, 20, 2)
    passport = st.sidebar.selectbox("Passport", ["Yes", "No"])
    pitch_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.sidebar.selectbox("Owns a Car", ["Yes", "No"])
    children = st.sidebar.slider("Number of Children Visiting", 0, 5, 1)
    designation = st.sidebar.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
    income = st.sidebar.number_input("Monthly Income", value=25000)
    duration = st.sidebar.number_input("Duration of Pitch", value=15)

    # --- Data Processing ---
    # Map inputs to internal encoding used during training
    # (Note: In a production app, these maps should match the LabelEncoder classes)
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [person_visiting],
        'NumberOfFollowups': [followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [trips],
        'Passport': [1 if passport == "Yes" else 0],
        'PitchSatisfactionScore': [pitch_score],
        'OwnCar': [1 if own_car == "Yes" else 0],
        'NumberOfChildrenVisiting': [children],
        'Designation': [designation],
        'MonthlyIncome': [income]
    })

    # The model pipeline handles Scaling and Encoding internally via Preprocessor if set up,
    # but since we used LabelEncoder manually in the script, we must replicate that logic here.
    # For this deployment app, we assume the pipeline's first step (StandardScaler) is ready.

    if st.button("Predict Purchase Likelihood"):
        try:
            # The model was trained on 18 features (after dropping Unnamed/CustomerID)
            # We need to ensure categorical columns are encoded as integers for the XGBoost model
            # This is a simplified encoding for the UI demo:
            encoding_map = {
                'TypeofContact': {'Self Enquiry': 1, 'Company Invited': 0},
                'Occupation': {'Salaried': 3, 'Small Business': 2, 'Large Business': 1, 'Free Lancer': 0},
                'Gender': {'Male': 1, 'Female': 0},
                'ProductPitched': {'Basic': 0, 'Deluxe': 1, 'Standard': 2, 'Super Deluxe': 3, 'King': 4},
                'MaritalStatus': {'Single': 3, 'Married': 1, 'Divorced': 0, 'Unmarried': 2},
                'Designation': {'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}
            }

            for col, mapping in encoding_map.items():
                input_data[col] = input_data[col].map(mapping)

            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            st.subheader("Prediction Results")
            if prediction[0] == 1:
                st.success(f"Target this customer! Probability of purchase: {probability:.2%}")
            else:
                st.warning(f"Low likelihood of purchase. Probability: {probability:.2%}")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
