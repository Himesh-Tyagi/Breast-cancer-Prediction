
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load pre-trained model and scaler
with open('bc.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scbc.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

   

def predict_cancer(features):
   
    # Convert features to a numpy array and reshape to match the model's input shape
    features = np.array(features).reshape(1, -1)

    # Preprocess the input features using the scaler
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)  # Get prediction probabilities

    # Interpret the prediction
    prediction_label = 'Malignant' if prediction[0] == 1 else 'Benign'
    proba_malignant = prediction_proba[0][1]  # Probability of being malignant
    proba_benign = prediction_proba[0][0]  # Probability of being benign
    
    
    # Provide a detailed interpretation
    interpretation = {
        'Prediction': prediction_label ,
        'Probability of Malignant':proba_malignant,
        'Probability of Benign': proba_benign,
      
    }

    return interpretation

# Streamlit UI
st.title("Breast Cancer Prediction App")

st.write("This app predicts Breast Cancer using a machine learning model. It determines whether a breast mass is benign (B) or malignant (M) based on the measurements.")

st.sidebar.write("Select Lab Parametres :")

# Define the input fields
texture_mean = st.sidebar.slider('texture_mean', min_value=0.0, max_value=200.0, value=0.0, step=0.1)
smoothness_mean = st.sidebar.slider('smoothness_mean', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
compactness_mean = st.sidebar.slider('compactness_mean', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
concave_points_mean = st.sidebar.slider('concave_points_mean', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
symmetry_mean = st.sidebar.slider('symmetry_mean', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
fractal_dimension_mean = st.sidebar.slider('fractal_dimension_mean', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
texture_se = st.sidebar.slider('texture_se', min_value=0.0, max_value=200.0, value=0.0, step=0.1)
area_se = st.sidebar.slider('area_se', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
smoothness_se = st.sidebar.slider('smoothness_se', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
compactness_se = st.sidebar.slider('compactness_se', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
concavity_se = st.sidebar.slider('concavity_se', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
concave_points_se = st.sidebar.slider('concave_points_se', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
symmetry_se = st.sidebar.slider('symmetry_se', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
fractal_dimension_se = st.sidebar.slider('fractal_dimension_se', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
texture_worst = st.sidebar.slider('texture_worst', min_value=0.0, max_value=200.0, value=0.0, step=0.1)
area_worst = st.sidebar.slider('area_worst', min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
smoothness_worst = st.sidebar.slider('smoothness_worst', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
concavity_worst = st.sidebar.slider('concavity_worst', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
compactness_worst = st.sidebar.slider('compactness_worst', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
concave_points_worst = st.sidebar.slider('concave_points_worst', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
symmetry_worst = st.sidebar.slider('symmetry_worst', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
fractal_dimension_worst = st.sidebar.slider('fractal_dimension_worst', min_value=0.0, max_value=2.0, value=0.0, step=0.1)

# Gather features into a list
features = [
    texture_mean, smoothness_mean, compactness_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
    texture_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
    texture_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
]




# Prediction
if st.button('Predict'):
    result = predict_cancer(features)
    st.write(f'The prediction is: {result}')
