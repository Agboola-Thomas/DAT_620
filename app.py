import streamlit as st
import numpy as np
from prediction import predict
from sklearn.preprocessing import MinMaxScaler

# App title
st.title('Mobile Price Range Prediction')



# User inputs for features
col1, col2 = st.columns(2)

with col1:
    battery_power = st.slider('Battery Power (mAh)', 500, 8000, step=100)
    dual_sim = st.selectbox('Dual SIM', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    four_g = st.selectbox('4G Enabled', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    int_memory = st.slider('Internal Memory (GB)', 2, 512)
    
with col2:
    mobile_wt = st.slider('Mobile Weight (g)', 80, 250)
    n_cores = st.slider('Number of Cores', 1, 8)
    ram = st.slider('RAM (MB)', 500, 8000, step=100)
    touch_screen = st.selectbox('Touch Screen', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Create a DataFrame for input
input_data = np.array([[battery_power, dual_sim, four_g, int_memory, mobile_wt, n_cores, ram, touch_screen]])

# Load the scaler (use the same scaling process as used during training)
scaler = MinMaxScaler()

# Separate columns for normalization
features_to_normalize = ['battery_power', 'int_memory', 'mobile_wt', 'n_cores', 'ram']
normalized_features = input_data[:, [0, 3, 4, 5, 6]]
non_normalized_features = input_data[:, [1, 2, 7]]

# Normalize the features
normalized_data = np.hstack([
    scaler.fit_transform(normalized_features),
    non_normalized_features
])

# Prediction
if st.button('Predict Price Range'):
    prediction = predict(normalized_data)

    # Determine the prediction text based on the prediction value
    if prediction[0] == 0:
        prediction_text = "Low Price Range"
    elif prediction[0] == 1:
        prediction_text = "Medium-Low Price Range"
    elif prediction[0] == 2:
        prediction_text = "Medium-High Price Range"
    else:
        prediction_text = "High Price Range"
    
    st.success(f'The predicted price range is: {prediction_text}')
