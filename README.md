# AQI-RELATED-HEALTH-ISSUES
This project predicts health conditions using features like age, existing health issues, exposure years, physical activity level, air quality index (AQI), and gender. It employs Random Forest classifiers for prediction. Users input personal data for tailored predictions, supporting early detection and personalized healthcare.


**Project Overview:**
The goal of this project is to develop predictive models for identifying the probability of different health conditions such as heart attack, stroke, blood pressure issues, lung cancer, eczema, psoriasis, bronchitis, COPD (Chronic Obstructive Pulmonary Disease), and asthma. The models utilize features such as age, existing health issues, exposure years, physical activity level, air quality index (AQI), and gender to make these predictions.

**Approach:**
1. **Data Preprocessing:** The dataset containing information on health conditions and associated features is loaded and preprocessed. Categorical variables are encoded, and percentages are converted to binary labels based on predefined thresholds.
  
2. **Model Training:** Random Forest classifiers are trained for each health condition using the preprocessed dataset. The number of trees in the forest is set to 100, and the models are trained on the features to predict the binary outcome of each health condition.

3. **User Input and Prediction:** After training the models, the program prompts the user to input their personal information including age, existing health issues, exposure years, physical activity level, AQI, and gender. The trained models then use these inputs to predict the likelihood of each health condition for the user.

**Result:**
The program provides predictions for each health condition based on the user's input, indicating whether the individual is likely to have the respective health condition or not.

Overall, this project demonstrates the application of machine learning techniques to healthcare by providing personalized predictions for various health conditions, thereby aiding in early detection and prevention efforts.
