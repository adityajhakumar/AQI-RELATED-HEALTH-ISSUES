import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# Load the dataset
df = pd.read_excel('health_data_percentages.xlsx')

# Encode categorical variables
label_encoder = LabelEncoder()
df['Physical_Activity_Level'] = label_encoder.fit_transform(df['Physical_Activity_Level'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Define features and target variables
X = df.drop(['Heart_Attack_Percentage', 'Stroke_Percentage', 'Blood_Pressure_Percentage',
              'Lung_Cancer_Percentage', 'Eczema_Percentage', 'Psoriasis_Percentage',
              'Bronchitis_Percentage', 'COPD_Percentage', 'Asthma_Percentage'], axis=1)

y = df[['Heart_Attack_Percentage', 'Stroke_Percentage', 'Blood_Pressure_Percentage',
        'Lung_Cancer_Percentage', 'Eczema_Percentage', 'Psoriasis_Percentage',
        'Bronchitis_Percentage', 'COPD_Percentage', 'Asthma_Percentage']]

# Train a Random Forest model for each health condition
models = {}
for disease in y.columns:
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y[disease])
    models[disease] = model

# Streamlit app
st.set_page_config(page_title="Tervive Health Prediction", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #ffffff;  /* White background */
        color: #004d40;  /* Dark green text */
    }
    .sidebar .sidebar-content {
        background-color: #e0f2f1;  /* Light green sidebar */
    }
    .stButton>button {
        background-color: #004d40;  /* Dark green button */
        color: white;  /* White text on button */
    }
    .stButton>button:hover {
        background-color: #003d33;  /* Darker green on hover */
    }
    </style>
    """, unsafe_allow_html=True)

# Display company name and title
st.title("TERVIVE")
st.subheader("Health Condition Prediction")

# Subtitle
st.markdown("### Health Risk Predictor")

# Add text about the relationship between diseases and air quality
st.markdown("""
**Understanding the Impact of Air Quality on Health**

Air pollution is a significant health concern with far-reaching effects. Prolonged exposure to poor air quality can lead to a range of health conditions:

- **Cardiovascular Diseases:** Pollutants can cause inflammation and damage to blood vessels, increasing the risk of heart disease and stroke.
- **Cancer:** Prolonged exposure to pollutants, particularly particulate matter and carcinogens, can elevate the risk of various cancers, including lung cancer.
- **Respiratory Conditions:** Pollutants exacerbate conditions like asthma, bronchitis, and COPD by irritating and inflaming the respiratory tract.
- **Skin Diseases:** Pollutants can contribute to skin issues such as eczema and psoriasis by damaging the skin barrier and increasing sensitivity to environmental triggers.

**Planting Trees to Combat Pollution**

Incorporating greenery into your surroundings can significantly help reduce air pollution:

- **Indoor Plants:** 
  - **Spider Plant (Chlorophytum comosum):** Known for its air-purifying qualities, it helps remove pollutants like formaldehyde.
  - **Peace Lily (Spathiphyllum):** Effective at removing airborne toxins and improving indoor air quality.
  - **Boston Fern (Nephrolepis exaltata):** Helps in filtering out formaldehyde and provides humidity.

- **Outdoor Trees:**
  - **Neem Tree (Azadirachta indica):** Effective in improving air quality and has anti-pollutant properties.
  - **Bamboo Palm (Chamaedorea seifrizii):** Known for its ability to filter out benzene, formaldehyde, and trichloroethylene.
  - **Golden Pothos (Epipremnum aureum):** Efficient in removing airborne toxins and enhancing indoor air quality.
""")

# Collect user input
age = st.number_input("Enter your age:", min_value=0, value=30)
existing_health_issues = st.selectbox("Do you have existing health issues?", [0, 1])
physical_activity_level = st.selectbox("Select your physical activity level", [0, 1, 2])
gender = st.selectbox("Select your gender", [0, 1])
initial_aqi = st.number_input("Enter the initial AQI (Air Quality Index):", min_value=0, value=100)

# Generate AQI values and calculate average AQI
aqi_values = np.arange(initial_aqi, initial_aqi + 20 * 2 + 1, 2)
avg_aqi = np.mean(aqi_values)

# Prepare input for prediction
input_data = pd.DataFrame({
    'Age': [age] * 20,
    'Existing_Health_Issues': [existing_health_issues] * 20,
    'Physical_Activity_Level': [physical_activity_level] * 20,
    'Gender': [gender] * 20,
    'Exposure_Years': list(range(1, 21)),
    'AQI': avg_aqi
})

# Make predictions using the trained models
predictions = {}
for disease, model in models.items():
    prediction = model.predict(input_data[X.columns])
    predictions[disease] = prediction

# Plot the predicted percentages for each health condition
fig, ax = plt.subplots(figsize=(10, 6))
for disease, prediction in predictions.items():
    ax.plot(range(1, 21), prediction, label=disease)

ax.set_title(f"Predicted Percentages for AQI = {avg_aqi}", fontsize=14)
ax.set_xlabel("Exposure Years", fontsize=12)
ax.set_ylabel("Percentage", fontsize=12)
ax.legend()
ax.grid(True)

# Save plot to a BytesIO object and display it in Streamlit
buf = BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)  # Rewind the buffer to the beginning
st.image(buf, caption="Predicted Health Percentages Over Exposure Years")

# Pie chart for predicted health conditions
fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
disease_labels = list(predictions.keys())
disease_values = [prediction[0] for prediction in predictions.values()]
ax_pie.pie(disease_values, labels=disease_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(disease_labels))))
ax_pie.set_title("Health Condition Distribution Based on Predictions")

# Save pie chart to a BytesIO object and display it in Streamlit
buf_pie = BytesIO()
plt.savefig(buf_pie, format="png")
buf_pie.seek(0)  # Rewind the buffer to the beginning
st.image(buf_pie, caption="Health Condition Distribution")

# Print the predicted percentages
st.subheader("Predicted Percentages for the First Year of Exposure:")
for disease, percentage in predictions.items():
    st.write(f"{disease}: {percentage[0]:.2f}%")
