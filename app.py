import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Function to load the model
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to preprocess the data
def preprocess_data(data, numerical_cols, categorical_cols, imputer_num, scaler, imputer_cat, encoder):
    data[numerical_cols] = imputer_num.transform(data[numerical_cols])
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    data[categorical_cols] = imputer_cat.transform(data[categorical_cols])
    encoded_cat = pd.DataFrame(encoder.transform(data[categorical_cols]), index=data.index)
    encoded_cat.columns = encoder.get_feature_names_out(categorical_cols)
    data = data.drop(categorical_cols, axis=1)
    data = pd.concat([data, encoded_cat], axis=1)
    return data

# Load the model
model = load_model()

# Load preprocessors
with open('imputer_num.pkl', 'rb') as f:
    imputer_num = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('imputer_cat.pkl', 'rb') as f:
    imputer_cat = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Define columns
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Streamlit app
st.title('House Price Prediction')

# Choose input method
input_method = st.radio("Choose input method", ("Upload CSV", "Input Form"))

if input_method == "Upload CSV":
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data", data.head())

        # Preprocess the data
        preprocessed_data = preprocess_data(data, numerical_cols, categorical_cols, imputer_num, scaler, imputer_cat, encoder)

        # Ensure the new data has the same columns as the training data
        preprocessed_data = preprocessed_data.reindex(columns=numerical_cols + encoder.get_feature_names_out(categorical_cols).tolist(), fill_value=0)

        # Make predictions
        predictions = model.predict(preprocessed_data)

        # Show predictions
        st.write("### Predictions", pd.DataFrame(predictions, columns=["Predicted Price"]))

elif input_method == "Input Form":
    st.write("### Input Form")

    area = st.number_input('Area', min_value=0)
    bedrooms = st.number_input('Bedrooms', min_value=0)
    bathrooms = st.number_input('Bathrooms', min_value=0)
    stories = st.number_input('Stories', min_value=0)
    parking = st.number_input('Parking', min_value=0)
    mainroad = st.selectbox('Mainroad', ['yes', 'no'])
    guestroom = st.selectbox('Guestroom', ['yes', 'no'])
    basement = st.selectbox('Basement', ['yes', 'no'])
    hotwaterheating = st.selectbox('Hotwater Heating', ['yes', 'no'])
    airconditioning = st.selectbox('Airconditioning', ['yes', 'no'])
    prefarea = st.selectbox('Prefarea', ['yes', 'no'])
    furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'mainroad': [mainroad],
            'guestroom': [guestroom],
            'basement': [basement],
            'hotwaterheating': [hotwaterheating],
            'airconditioning': [airconditioning],
            'parking': [parking],
            'prefarea': [prefarea],
            'furnishingstatus': [furnishingstatus]
        })

        # Preprocess the data
        preprocessed_data = preprocess_data(input_data, numerical_cols, categorical_cols, imputer_num, scaler, imputer_cat, encoder)

        # Ensure the new data has the same columns as the training data
        preprocessed_data = preprocessed_data.reindex(columns=numerical_cols + encoder.get_feature_names_out(categorical_cols).tolist(), fill_value=0)

        # Make predictions
        predictions = model.predict(preprocessed_data)

        # Show predictions
        st.write("### Prediction", pd.DataFrame(predictions, columns=["Predicted Price"]))