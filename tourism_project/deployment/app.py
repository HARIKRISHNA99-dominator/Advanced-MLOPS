import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load the trained model from Hugging Face Model Hub
# hf_hub_download works inside Docker; a local file path would not
model_path = hf_hub_download(
    repo_id="carnage-colossus/tourism-prediction-model",
    filename="best_tourism_prediction_model_v1.joblib"
)
model = joblib.load(model_path)

st.set_page_config(page_title="Tourism Package Prediction", layout="wide")
st.title("🌐 Visit with Us: Wellness Package Predictor")
st.markdown("""
This app predicts whether a customer is likely to purchase the new
**Wellness Tourism Package** based on their profile and interaction history.
""")

st.sidebar.header("Customer Attributes")

# --- Input widgets ---
age             = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
# Values must match the exact strings in the dataset (LabelEncoder alphabetical order)
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier       = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation      = st.sidebar.selectbox("Occupation",
                    ["Free Lancer", "Large Business", "Salaried", "Small Business"])
# 'Fe Male' exists in the dataset — must be included
gender          = st.sidebar.selectbox("Gender", ["Fe Male", "Female", "Male"])
person_visiting = st.sidebar.slider("Number of Persons Visiting", 1, 10, 3)
followups       = st.sidebar.slider("Number of Follow-ups", 1, 10, 3)
product_pitched = st.sidebar.selectbox("Product Pitched",
                    ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
property_star   = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
marital_status  = st.sidebar.selectbox("Marital Status",
                    ["Divorced", "Married", "Single", "Unmarried"])
trips           = st.sidebar.number_input("Number of Trips", 1, 20, 2)
passport        = st.sidebar.selectbox("Passport", ["Yes", "No"])
pitch_score     = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car         = st.sidebar.selectbox("Owns a Car", ["Yes", "No"])
children        = st.sidebar.slider("Number of Children Visiting", 0, 5, 1)
designation     = st.sidebar.selectbox("Designation",
                    ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
income          = st.sidebar.number_input("Monthly Income", value=25000)
duration        = st.sidebar.number_input("Duration of Pitch (mins)", value=15)

# --- Encoding maps (alphabetical = LabelEncoder default order) ---
# These must exactly match the encoding applied in prep.py
contact_map  = {"Company Invited": 0, "Self Enquiry": 1}
occ_map      = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
gender_map   = {"Fe Male": 0, "Female": 1, "Male": 2}
product_map  = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
marital_map  = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
desig_map    = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

if st.button("Predict Purchase Likelihood"):
    input_data = pd.DataFrame([{
        "Age":                     age,
        "TypeofContact":           contact_map[type_of_contact],
        "CityTier":                city_tier,
        "DurationOfPitch":         duration,
        "Occupation":              occ_map[occupation],
        "Gender":                  gender_map[gender],
        "NumberOfPersonVisiting":  person_visiting,
        "NumberOfFollowups":       followups,
        "ProductPitched":          product_map[product_pitched],
        "PreferredPropertyStar":   property_star,
        "MaritalStatus":           marital_map[marital_status],
        "NumberOfTrips":           trips,
        "Passport":                1 if passport == "Yes" else 0,
        "PitchSatisfactionScore":  pitch_score,
        "OwnCar":                  1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting":children,
        "Designation":             desig_map[designation],
        "MonthlyIncome":           income,
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Target this customer! Purchase probability: {probability:.2%}")
    else:
        st.warning(f"Low likelihood of purchase. Probability: {probability:.2%}")
