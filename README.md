# HousePrediction


This project uses the California Housing dataset to predict median house prices for different districts using Machine Learning. The goal is to build a regression model that can accurately estimate house prices based on features like income, location, and number of rooms.

The entire process includes:

Data downloading & loading

Data exploration & visualization

Feature engineering & preprocessing

Model training & evaluation

Hyperparameter tuning (GridSearchCV)

Final performance testing with confidence intervals

🎯 Objective
Predict house prices using:

🏘️ Median income

📅 Housing age

🛏️ Total rooms

🌊 Ocean proximity (categorical)

👨‍👩‍👧‍👦 Population & household data

We use these features to train regression models and evaluate which one gives the most accurate price predictions.

📦 Dataset
📁 Source: California Housing Dataset

📊 Features: Median income, rooms, location, population, etc.

🎯 Target: median_house_value (the value we want to predict)

🛠️ ML Tools & Libraries
pandas, numpy – Data manipulation

matplotlib, seaborn – Visualizations

scikit-learn – ML models and preprocessing

GridSearchCV – Hyperparameter tuning

RandomForestRegressor – Best performing model

🔄 Workflow Overview
1️⃣ Data Loading
Downloaded from GitHub and extracted

Loaded using pandas.read_csv()

2️⃣ Data Exploration
Histograms and scatter plots

Correlation analysis

Geographical visualizations (longitude vs latitude)

3️⃣ Feature Engineering
Created new features:

rooms_per_household

population_per_household

bedrooms_per_room

4️⃣ Preprocessing Pipeline
Handled missing values using SimpleImputer

One-hot encoded categorical data (ocean_proximity)

Feature scaling with StandardScaler

Custom CombinedAttributesAdder transformer added engineered features

5️⃣ Model Training & Evaluation
Trained three models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor ✅ (Best performance)

Evaluated using:

Training RMSE

Cross-validation

Final test set performance

6️⃣ Hyperparameter Tuning
Tuned RandomForestRegressor with GridSearchCV

Evaluated top features using feature importances

7️⃣ Final Evaluation
Used the best model from GridSearchCV

Calculated:

RMSE on the test set

95% Confidence Interval using scipy.stats

📈 Results
Model	RMSE (Cross-Validation)
Linear Regression	High error
Decision Tree	Overfits
✅ Random Forest	Best accuracy and generalization

🤖 How to Use
Prepare new housing data in the same format.

Run it through the full_pipeline for preprocessing.

Use the trained model:

python
Copy
Edit
prediction = final_model.predict(prepared_data)
📂 Project Structure
bash
Copy
Edit
📁 housing-price-predictor
│
├── datasets/
│   └── housing/
│       └── housing.csv
├── housing_model.py  # Main ML script
├── README.md         # This file
└── requirements.txt  # Libraries used
📌 Future Improvements
Add XGBoost or LightGBM for even better performance

Deploy using Streamlit or Flask

Save and load models using joblib

Add interactive visualization dashboards

🧠 Key Learnings
Hands-on experience with a full ML pipeline

Feature engineering really boosts performance

Importance of cross-validation and model tuning

Pipelines help make your code modular and reusable


