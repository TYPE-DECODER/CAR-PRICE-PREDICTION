import streamlit as st
import pickle
import numpy as np

# Load the model
file = open('random_forest_regression_model.pkl', 'rb')
model = pickle.load(file)

def main():
    st.title('Car Price Prediction')
    
    # Create input fields for the features
    car_name = st.text_input('Car Name')
    year = st.number_input('Year', min_value=1900, max_value=2023, value=2015)
    present_price = st.number_input('Present Price (in lakhs)', min_value=0.0, max_value=100.0, value=0.0)
    kms_driven = st.number_input('Kilometers Driven', min_value=0, max_value=1000000, value=50000)
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner', [0, 1, 2, 3])

    # Preprocess the input features
    no_year = 2023 - year

    fuel_type_diesel = 1 if fuel_type == 'Diesel' else 0
    fuel_type_petrol = 1 if fuel_type == 'Petrol' else 0

    seller_type_individual = 1 if seller_type == 'Individual' else 0
    transmission_manual = 1 if transmission == 'Manual' else 0

    # Prepare the feature array for prediction
    features = np.array([[present_price, kms_driven, owner, no_year, fuel_type_diesel, fuel_type_petrol, seller_type_individual, transmission_manual]])
    
    # Predict the car price
    if st.button('Predict Price'):
        predicted_price = model.predict(features)
        st.success(f'The predicted selling price of the car is {predicted_price[0]:.2f} lakhs')

if __name__ == '__main__':
    main()
