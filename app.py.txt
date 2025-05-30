import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
 
# Load the model
model = joblib.load('ann_model.pkl')
 
# Define feature options

restecg_options = {
    0: 'No (0)',
    1: 'Somewhat (1)',
    2: 'Very (2)'
}
 
slope_options = {
    1: 'Poor (1)',
    2: 'Fair (2)',
    3: 'Good (3)'
}
 
 
# Define feature names
feature_names = [
    "Sex", "Life Satisfication", "Digestive_Disease", "Brain Damage",
    "Depression", "Familysize", "Self Health Asses", "Pain"
]
 
# Streamlit user interface
st.title("COPD Insomnia Risk Predictor")
 

# Sex: categorical selection
Sex = st.selectbox("Sex (1=Male,2=Female, ):", options=[1, 2], format_func=lambda x: 'Male (1) if x == 1 else 'Female (2)'')

# life_satisfication: categorical selection
life_satisfication = st.selectbox("Life Satisfication:", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

# Stomach_or_other_digestive_disease: categorical selection
Stomach_or_other_digestive_disease = st.selectbox("Digestive_Disease:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Brain_damage: categorical selection
Brain_damage = st.selectbox("Brain Damage:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# depression: Depression
depression= st.selectbox("Depression:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
 
 
# Familysize: numerical input
Familysize = st.number_input("Familysize:", min_value=1, max_value=13, value=4)
 

# self_health: categorical selection
self_health = st.selectbox("Self Health Asses:", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
 
# Pain: categorical selection
Pain = st.selectbox("Pain:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
 
# Process inputs and make predictions
feature_values = [Sex, life_satisfication, Stomach_or_other_digestive_disease, Brain_damage, depression, Familysize, self_health, Pain]
features = np.array([feature_values])
 
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
 
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
 
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
 
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )
 
    st.write(advice)
 
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
 
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
 
    st.image("shap_force_plot.png")