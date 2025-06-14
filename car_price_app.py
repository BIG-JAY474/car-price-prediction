import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------- Simulate data (same as before) --------
np.random.seed(1)
mileage = np.random.randint(0, 200000, size=100)
price = 30000 - (0.1 * mileage) + np.random.normal(0, 1000, 100)

df = pd.DataFrame({'mileage': mileage, 'price': price})

# -------- Train model --------
model = LinearRegression()
model.fit(df[['mileage']], df['price'])

# -------- Streamlit UI --------
st.title("🚗 Car Price Predictor")
st.write("Enter your car's mileage to estimate its market price.")

mileage_input = st.number_input("Car Mileage (km)", min_value=0, max_value=300000, value=50000, step=500)

if st.button("Predict Price"):
    input_df = pd.DataFrame({'mileage': [mileage_input]})
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
