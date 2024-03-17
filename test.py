import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

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

# Train a Random Forest model for each health condition with bagging
models = {}
for disease in y.columns:
    model = RandomForestRegressor(n_estimators=100, random_state=0)  # Set number of trees to 100
    model.fit(X, y[disease])
    models[disease] = model

# Collect user input for testing
age = int(input("Enter your age: "))
existing_health_issues = int(input("Enter 1 if you have existing health issues, 0 otherwise: "))
physical_activity_level = int(input("Enter your physical activity level (0 for Low, 1 for Moderate, 2 for High): "))
gender = int(input("Enter 1 for Male, 0 for Female: "))

# Generate AQI values
initial_aqi = int(input("Enter the initial AQI (Air Quality Index): "))
aqi_values = np.arange(initial_aqi, initial_aqi + 20 * 2 + 1, 2)

# Calculate average AQI value
avg_aqi = np.mean(aqi_values)

# Prepare input for prediction
input_data = pd.DataFrame({
    'Age': [age] * 20,
    'Existing_Health_Issues': [existing_health_issues] * 20,
    'Physical_Activity_Level': [physical_activity_level] * 20,
    'Gender': [gender] * 20,
    'Exposure_Years': list(range(1, 21)),
    'AQI': avg_aqi  # Use the average AQI value for all exposure years
})

# Make predictions using the trained models
predictions = {}
for disease, model in models.items():
    prediction = model.predict(input_data[X.columns])
    predictions[disease] = prediction

# Plot the predicted percentages for each health condition
plt.figure(figsize=(10, 6))
for disease, prediction in predictions.items():
    plt.plot(range(1, 21), prediction, label=disease)

plt.title(f"Predicted Percentages for AQI = {avg_aqi}")
plt.xlabel("Exposure Years")
plt.ylabel("Percentage")
plt.legend()
plt.grid(True)
plt.show()

# Print the predicted percentages
for disease, percentage in predictions.items():
    print(f"{disease}: {percentage[0]:.2f}%")
