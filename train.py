import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the new dataset
df = pd.read_excel('health_data_percentages.xlsx')

# Encode categorical variables if needed
label_encoder = LabelEncoder()
df['Physical_Activity_Level'] = label_encoder.fit_transform(df['Physical_Activity_Level'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Define features
X = df.drop(['Heart_Attack_Percentage', 'Stroke_Percentage', 'Blood_Pressure_Percentage', 'Lung_Cancer_Percentage', 'Eczema_Percentage', 'Psoriasis_Percentage', 'Bronchitis_Percentage', 'COPD_Percentage', 'Asthma_Percentage'], axis=1)

# Define thresholds for classification
thresholds = {
    'Heart_Attack_Percentage': 0.5,  # Example threshold, adjust as needed
    'Stroke_Percentage': 0.5,  # Example threshold, adjust as needed
    'Blood_Pressure_Percentage': 0.5,  # Example threshold, adjust as needed
    'Lung_Cancer_Percentage': 0.5,  # Example threshold, adjust as needed
    'Eczema_Percentage': 0.5,  # Example threshold, adjust as needed
    'Psoriasis_Percentage': 0.5,  # Example threshold, adjust as needed
    'Bronchitis_Percentage': 0.5,  # Example threshold, adjust as needed
    'COPD_Percentage': 0.5,  # Example threshold, adjust as needed
    'Asthma_Percentage': 0.5  # Example threshold, adjust as needed
}

# Convert percentages to binary labels based on thresholds
for disease, threshold in thresholds.items():
    df[disease] = (df[disease] > threshold).astype(int)

# Define target variables
y = df[['Heart_Attack_Percentage', 'Stroke_Percentage', 'Blood_Pressure_Percentage', 'Lung_Cancer_Percentage', 'Eczema_Percentage', 'Psoriasis_Percentage', 'Bronchitis_Percentage', 'COPD_Percentage', 'Asthma_Percentage']]

# Train a Random Forest model for each health condition
models = {}
for disease in y.columns:
    model = RandomForestClassifier(n_estimators=100, random_state=0)  # Set number of trees to 100
    model.fit(X, y[disease])
    models[disease] = model

# Evaluate the accuracy of each model using cross-validation
accuracies = {}
for disease, model in models.items():
    scores = cross_val_score(model, X, y[disease], cv=5, scoring='accuracy')
    accuracies[disease] = np.mean(scores)

# Print the accuracy of each model
print("Accuracy of each model:")
for disease, accuracy in accuracies.items():
    print(f"{disease}: {accuracy:.2f}")
