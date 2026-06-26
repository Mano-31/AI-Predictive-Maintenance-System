import streamlit as st
import pandas as pd
import joblib


# Page Settings
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🏭",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# CSS - Laptop Screen Size
st.markdown("""
<style>

.block-container {
    max-width: 700px;
    padding-top: 1rem;
    padding-bottom: 1rem;
}

h1 {
    font-size: 32px !important;
    text-align: center;
}

.stNumberInput input {
    font-size: 16px;
}

button {
    width: 100%;
    height: 45px;
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)



# Load Model
model = joblib.load(
    "models/predictive_maintenance_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)



# Title

st.markdown(
    "<h1>🏭 AI-Based Predictive Maintenance System</h1>",
    unsafe_allow_html=True
)

st.write(
    "Predict machine failure using Machine Learning"
)



# User Input

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)


air_temp = st.number_input(
    "Air Temperature [K]",
    value=300.0
)


process_temp = st.number_input(
    "Process Temperature [K]",
    value=310.0
)


rpm = st.number_input(
    "Rotational Speed [rpm]",
    value=1500
)


torque = st.number_input(
    "Torque [Nm]",
    value=40.0
)


tool_wear = st.number_input(
    "Tool Wear [min]",
    value=100
)



# Encoding

type_map = {
    "L":0,
    "M":1,
    "H":2
}



# Feature Engineering

temp_difference = process_temp - air_temp

power = torque * rpm



# Input Data

input_data = pd.DataFrame({

    "UDI":[1],

    "Type":[type_map[machine_type]],

    "Air_temperature_K":[air_temp],

    "Process_temperature_K":[process_temp],

    "Rotational_speed_rpm":[rpm],

    "Torque_Nm":[torque],

    "Tool_wear_min":[tool_wear],

    "TWF":[0],

    "HDF":[0],

    "PWF":[0],

    "OSF":[0],

    "RNF":[0],

    "Temperature_Difference":[temp_difference],

    "Power":[power]

})



# Prediction

if st.button("Predict Failure"):


    scaled_data = scaler.transform(
        input_data
    )


    prediction = model.predict(
        scaled_data
    )



    if prediction[0] == 1:

        st.error(
            "⚠️ Machine Failure Risk Detected"
        )


    else:

        st.success(
            "✅ Machine Working Normally"
        )