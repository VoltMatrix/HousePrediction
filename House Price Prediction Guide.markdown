# House Price Prediction Using Machine Learning

## Introduction
This document explains how to build a machine learning model to predict house prices using the California Housing dataset. The goal is to create a model that accurately estimates the median house value for a district based on features like median income, number of rooms, and location. We’ll walk through downloading the data, exploring it, preprocessing it, training multiple models, and optimizing the best one. Each step is explained in plain language, with detailed comments in the code to ensure clarity and preserve knowledge for future use.

## Objective
The objective is to predict house prices (a continuous value) using a dataset containing features like:
- **Median income**: Income of residents in the district.
- **Housing age**: Median age of houses.
- **Total rooms**: Number of rooms in the district.
- **Ocean proximity**: How close the district is to the ocean (categorical: e.g., "NEAR OCEAN").
- **Population, households, etc.**: Other demographic and housing data.

We use regression models (Linear Regression, Decision Tree, Random Forest) to learn patterns from the data and predict house prices. The code includes data preprocessing, visualization, model training, and hyperparameter tuning to achieve the best predictions.

## Dataset
We use the California Housing dataset, which contains housing data from the 1990 California census. It includes numerical features (e.g., median income) and one categorical feature (ocean proximity). The target variable is `median_house_value`, the price we aim to predict.

## Step-by-Step Explanation
Below, we break down the code into sections, explaining what each part does and how it contributes to building the house price predictor. Each code block includes detailed comments explaining every line, ensuring the knowledge is clear and reusable.

---

### 1. Importing Libraries
**What this code does**: This section imports all the Python libraries needed for the project, including tools for data handling, visualization, preprocessing, and machine learning. These libraries provide functions to process data, build models, and evaluate their performance, which are essential for predicting house prices accurately.

**How it relates to the goal**: Without these libraries, we couldn’t load the dataset, clean it, visualize patterns, or train models. Each library serves a specific purpose in the pipeline to achieve accurate predictions.

```python
# Import libraries for various tasks in the machine learning pipeline
import os  # Handles file paths and directories (e.g., creating folders for data)
import tarfile  # Extracts compressed .tgz files (dataset is downloaded in this format)
import urllib.request  # Downloads files from the internet (to fetch the dataset)
import pandas as pd  # Manages data in tabular format (DataFrames) for easy manipulation
import numpy as np  # Performs numerical computations (e.g., arrays, math operations)
import matplotlib.pyplot as plt  # Creates visualizations (e.g., histograms, scatter plots) to explore data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit  # Splits data into training and test sets; StratifiedShuffleSplit ensures balanced splits
from sklearn.impute import SimpleImputer  # Fills missing values in the dataset (e.g., with median)
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder  # Encodes categorical variables (e.g., ocean proximity) into numerical format
from sklearn.base import BaseEstimator, TransformerMixin  # Enables creation of custom data transformers (e.g., for feature engineering)
from sklearn.pipeline import Pipeline  # Chains multiple preprocessing steps for streamlined processing
from sklearn.preprocessing import StandardScaler  # Scales numerical features to have mean=0, std=1 for better model performance
from sklearn.compose import ColumnTransformer  # Applies different transformations to numerical and categorical columns
from sklearn.linear_model import LinearRegression  # Builds a linear regression model for prediction
from sklearn.metrics import mean_squared_error  # Evaluates model performance by calculating error
from sklearn.tree import DecisionTreeRegressor  # Builds a decision tree model for prediction
from sklearn.model_selection import cross_val_score  # Performs cross-validation to assess model robustness
from sklearn.ensemble import RandomForestRegressor  # Builds a random forest model, an ensemble of decision trees
from sklearn.model_selection import GridSearchCV  # Tunes hyperparameters to optimize model performance
from scipy import stats  # Computes statistical measures (e.g., confidence intervals for errors)
```

---

### 2. Downloading the Dataset
**What this code does**: This section defines the URL and local path for the California Housing dataset, downloads it as a compressed `.tgz` file, and extracts it to a local directory. The dataset is then loaded into a pandas DataFrame for analysis.

**How it relates to the goal**: The dataset contains the features and target variable (house prices) needed to train the model. Downloading and loading it correctly is the first step to building the predictor.

```python
# Define constants for dataset download
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"  # Base URL where dataset is hosted
HOUSING_PATH = os.path.join("datasets", "housing")  # Local directory to store dataset (e.g., 'datasets/housing')
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"  # Full URL to the compressed dataset file

# Function to download and extract the housing dataset
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):  # Check if the local directory exists
        os.makedirs(housing_path)  # Create the directory if it doesn’t exist
    tgz_path = os.path.join(housing_path, "housing.tgz")  # Path for the downloaded .tgz file
    urllib.request.urlretrieve(housing_url, tgz_path)  # Download the compressed file from the URL
    with tarfile.open(tgz_path) as housing_tgz:  # Open the .tgz file
        housing_tgz.extractall(path=housing_path)  # Extract the CSV file to the local directory

# Download and extract the dataset
fetch_housing_data()  # Call the function to download and unzip the dataset

# Function to load the dataset into a pandas DataFrame
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")  # Path to the extracted CSV file
    return pd.read_csv(csv_path)  # Load the CSV file into a pandas DataFrame

# Load the dataset and store it in a variable
housing = load_housing_data()  # Load the data into the 'housing' DataFrame
print(housing.head())  # Display the first 5 rows to inspect the dataset structure
print(housing.info())  # Show column names, data types, and missing value counts
```

---

### 3. Exploring the Data
**What this code does**: This section visualizes the distribution of numerical features (e.g., median income, house value) using histograms and checks the dataset’s structure. It also creates a temporary `income_cat` column to ensure the training and test sets are representative of the data’s income distribution.

**How it relates to the goal**: Exploring the data helps us understand its patterns (e.g., how income relates to house prices) and ensures the training/test split is balanced, which is critical for building a model that generalizes well to new data.

```python
# Visualize the distribution of numerical features
housing.hist(bins=50, figsize=(20,15))  # Create histograms for all numerical columns with 50 bins
plt.show()  # Display the histograms to understand data distributions (e.g., skewed or normal)

# Create income categories for stratified sampling
housing["income_cat"] = pd.cut(
    housing["median_income"],  # Convert continuous median_income into discrete bins
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],  # Define bin edges for income categories
    labels=[1, 2, 3, 4, 5]  # Assign integer labels to each bin (1=lowest, 5=highest)
)

# Perform stratified sampling to split data into training and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Initialize stratified split with 20% test size, fixed random seed
for train_index, test_index in split.split(housing, housing["income_cat"]):  # Split based on income_cat to maintain proportions
    strat_train_set = housing.loc[train_index]  # Create training set with selected indices
    strat_test_set = housing.loc[test_index]  # Create test set with selected indices

# Verify the proportion of income categories in the test set
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))  # Show proportions to confirm representativeness

# Remove the temporary income_cat column from both sets
for set_ in (strat_train_set, strat_test_set):  # Loop through training and test sets
    set_.drop("income_cat", axis=1, inplace=True)  # Drop income_cat as it’s no longer needed
```

---

### 4. Visualizing Geographical and Correlation Data
**What this code does**: This section creates scatter plots to visualize the geographical distribution of houses (based on latitude/longitude) and their relationship with population and house value. It also computes and visualizes correlations between features and the target variable (`median_house_value`).

**How it relates to the goal**: Visualizations reveal patterns, like how house prices vary by location or income, which helps us understand which features are important for prediction. Correlation analysis identifies which features (e.g., median income) strongly influence house prices.

```python
# Visualize geographical distribution of training data
strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)  # Plot locations with low opacity to show density
plt.show()  # Display the scatter plot to see where houses are concentrated

# Visualize population and house value geographically
strat_train_set.plot(
    kind="scatter", x="longitude", y="latitude", alpha=0.4,  # Plot locations with moderate opacity
    s=strat_train_set["population"]/100, label="population", figsize=(10,7),  # Circle size proportional to population
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True  # Color by house value, using jet colormap
)
plt.legend()  # Add a legend to the plot
plt.show()  # Display to see how population and house value vary by location

# Check correlations with the target variable
corr_matrix = strat_train_set.corr()  # Compute correlation matrix for numerical columns
print(corr_matrix["median_house_value"].sort_values(ascending=False))  # Show correlations with house value, sorted high to low

# Visualize correlations between key attributes
from pandas.plotting import scatter_matrix  # Import scatter matrix for pairwise plots
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]  # Select key features
scatter_matrix(strat_train_set[attributes], figsize=(12,8))  # Create scatter plots for selected features
plt.show()  # Display to inspect relationships (e.g., median_income vs. house value)

# Plot median income vs. house value
strat_train_set.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)  # Scatter plot with low opacity
plt.show()  # Display to confirm strong correlation between income and house value
```

---

### 5. Feature Engineering
**What this code does**: This section creates new features (e.g., rooms per household) to capture more meaningful relationships in the data. It then recomputes correlations to see if the new features improve prediction power.

**How it relates to the goal**: Adding features like rooms per household or bedrooms per room can make the model more accurate by providing more relevant information about each district, improving its ability to predict house prices.

```python
# Create a copy of the training set for feature engineering
housing = strat_train_set.copy()  # Copy to avoid modifying the original training set
# Add new features to capture meaningful relationships
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]  # Compute average rooms per household
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]  # Compute ratio of bedrooms to total rooms
housing["population_per_household"] = housing["population"] / housing["households"]  # Compute average population per household

# Check correlations with new features
corr_matrix = housing.corr()  # Recompute correlation matrix including new features
print(corr_matrix["median_house_value"].sort_values(ascending=False))  # Show updated correlations with house value
```

---

### 6. Data Preprocessing
**What this code does**: This section prepares the data for machine learning by handling missing values, encoding the categorical feature (`ocean_proximity`), and creating a custom transformer to add new features. It uses a pipeline to combine numerical and categorical processing steps.

**How it relates to the goal**: Preprocessing ensures the data is clean and in the right format for the models. Handling missing values, scaling numerical features, and encoding categorical variables make the data suitable for accurate predictions.

```python
# Prepare data for machine learning
housing = strat_train_set.drop("median_house_value", axis=1)  # Separate features by dropping target variable
housing_labels = strat_train_set["median_house_value"].copy()  # Store target variable (house prices)

# Handle missing values in numerical columns
imputer = SimpleImputer(strategy="median")  # Initialize imputer to fill missing values with column medians
housing_num = housing.drop("ocean_proximity", axis=1)  # Drop categorical column for numerical processing
imputer.fit(housing_num)  # Compute median for each numerical column
X = imputer.transform(housing_num)  # Replace missing values with medians
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)  # Convert back to DataFrame
print(housing_tr.info())  # Verify no missing values in numerical data

# Explore categorical variable (ocean_proximity)
housing_cat = housing[["ocean_proximity"]]  # Select the categorical column
print(housing_cat.head(10))  # Display first 10 rows to see categories (e.g., NEAR OCEAN, INLAND)

# Try ordinal encoding (for exploration, not used in final model)
ordinal_encoder = OrdinalEncoder()  # Initialize encoder to convert categories to numbers
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)  # Encode categories (e.g., NEAR OCEAN -> 0)
print(housing_cat_encoded[:10])  # Show first 10 encoded values

# Use one-hot encoding for categorical variable (preferred for models)
cat_encoder = OneHotEncoder()  # Initialize encoder to convert categories to binary columns
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)  # Encode categories into sparse matrix
print(housing_cat_1hot.toarray()[:10])  # Convert to array and show first 10 rows

# Custom transformer to add new features
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # Option to include bedrooms_per_room feature
        self.add_bedrooms_per_room = add_bedrooms_per_room  # Store parameter
    def fit(self, X, y=None):  # No parameters to learn, return self
        return self
    def transform(self, X):  # Add new features to the data
        col_names = housing_num.columns  # Get column names of numerical data
        rooms_ix = np.where(col_names == "total_rooms")[0][0]  # Find index of total_rooms column
        bedrooms_ix = np.where(col_names == "total_bedrooms")[0][0]  # Find index of total_bedrooms
        population_ix = np.where(col_names == "population")[0][0]  # Find index of population
        households_ix = np.where(col_names == "households")[0][0]  # Find index of households
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]  # Compute rooms per household
        population_per_household = X[:, population_ix] / X[:, households_ix]  # Compute population per household
        if self.add_bedrooms_per_room:  # If enabled, compute bedrooms per room
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]  # Compute ratio of bedrooms to rooms
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]  # Return data with all new features
        return np.c_[X, rooms_per_household, population_per_household]  # Return data with two new features

# Pipeline for numerical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),  # Step 1: Fill missing values with median
    ('attribs_adder', CombinedAttributesAdder()),  # Step 2: Add new features using custom transformer
    ('std_scaler', StandardScaler()),  # Step 3: Scale features to mean=0, std=1
])

# Define numerical and categorical columns
num_attribs = list(housing_num)  # List of numerical column names
cat_attribs = ["ocean_proximity"]  # List of categorical column names

# Full pipeline combining numerical and categorical processing
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),  # Apply numerical pipeline to numerical columns
    ("cat", OneHotEncoder(), cat_attribs),  # Apply one-hot encoding to categorical column
])

# Transform training data
housing_prepared = full_pipeline.fit_transform(housing)  # Apply full pipeline to prepare training data
```

---

### 7. Training and Evaluating Models
**What this code does**: This section trains three models (Linear Regression, Decision Tree, Random Forest) on the prepared data and evaluates their performance using RMSE (Root Mean Squared Error) on the training set and via cross-validation. It also tests the models on the test set.

**How it relates to the goal**: Training multiple models allows us to compare their performance and choose the best one for predicting house prices. Cross-validation and test set evaluation ensure the model generalizes well to new data.

```python
# Train Linear Regression model
lin_reg = LinearRegression()  # Initialize linear regression model
lin_reg.fit(housing_prepared, housing_labels)  # Train model on prepared data and target values

# Test predictions on a small sample
some_data = housing.iloc[:5]  # Select first 5 rows of training features
some_labels = housing_labels.iloc[:5]  # Select corresponding target values
some_data_prepared = full_pipeline.transform(some_data)  # Prepare sample using pipeline
print("Predictions:", lin_reg.predict(some_data_prepared))  # Predict house prices for sample
print("Labels:", list(some_labels))  # Show actual house prices for comparison

# Evaluate Linear Regression on training set
housing_predictions = lin_reg.predict(housing_prepared)  # Predict on entire training set
lin_mse = mean_squared_error(housing_labels, housing_predictions)  # Compute mean squared error
lin_rmse = np.sqrt(lin_mse)  # Compute root mean squared error
print("Linear Regression RMSE (train):", lin_rmse)  # Display RMSE to assess fit

# Train Decision Tree model
tree_reg = DecisionTreeRegressor(random_state=42)  # Initialize decision tree with fixed random seed
tree_reg.fit(housing_prepared, housing_labels)  # Train model on prepared data

# Evaluate Decision Tree on training set
housing_predictions = tree_reg.predict(housing_prepared)  # Predict on training set
tree_mse = mean_squared_error(housing_labels, housing_predictions)  # Compute MSE
tree_rmse = np.sqrt(tree_mse)  # Compute RMSE
print("Decision Tree RMSE (train):", tree_rmse)  # Display RMSE (likely 0 due to overfitting)

# Cross-validation for Decision Tree
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring="neg_mean_squared_error", cv=10)  # Perform 10-fold cross-validation
tree_rmse_scores = np.sqrt(-scores)  # Convert negative MSE to RMSE

# Function to display cross-validation scores
def display_scores(scores):
    print("Scores:", scores)  # Show RMSE for each fold
    print("Mean:", scores.mean())  # Show average RMSE
    print("Standard deviation:", scores.std())  # Show variability of RMSE

print("Decision Tree Cross-Validation:")
display_scores(tree_rmse_scores)  # Display cross-validation results

# Cross-validation for Linear Regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)  # Perform 10-fold cross-validation
lin_rmse_scores = np.sqrt(-lin_scores)  # Convert negative MSE to RMSE
print("Linear Regression Cross-Validation:")
display_scores(lin_rmse_scores)  # Display cross-validation results

# Train Random Forest model
forest_reg = RandomForestRegressor(random_state=42)  # Initialize random forest with fixed random seed
forest_reg.fit(housing_prepared, housing_labels)  # Train model on prepared data

# Cross-validation for Random Forest
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)  # Perform 10-fold cross-validation
forest_rmse_scores = np.sqrt(-forest_scores)  # Convert negative MSE to RMSE
print("Random Forest Cross-Validation:")
display_scores(forest_rmse_scores)  # Display cross-validation results

# Evaluate models on test set
test_data = strat_test_set.drop("median_house_value", axis=1)  # Separate test set features
test_labels = strat_test_set["median_house_value"].copy()  # Store test set target values
test_prepared = full_pipeline.transform(test_data)  # Prepare test set using pipeline

# Linear Regression on test set
test_predictions = lin_reg.predict(test_prepared)  # Predict on test set
test_mse = mean_squared_error(test_labels, test_predictions)  # Compute MSE
test_rmse = np.sqrt(test_mse)  # Compute RMSE
print("Linear Regression RMSE (test):", test_rmse)  # Display test set performance

# Decision Tree on test set
test_predictions = tree_reg.predict(test_prepared)  # Predict on test set
test_mse = mean_squared_error(test_labels, test_predictions)  # Compute MSE
test_rmse = np.sqrt(test_mse)  # Compute RMSE
print("Decision Tree RMSE (test):", test_rmse)  # Display test set performance

# Random Forest on test set
test_predictions = forest_reg.predict(test_prepared)  # Predict on test set
test_mse = mean_squared_error(test_labels, test_predictions)  # Compute MSE
test_rmse = np.sqrt(test_mse)  # Compute RMSE
print("Random Forest RMSE (test):", test_rmse)  # Display test set performance
```

---

### 8. Hyperparameter Tuning with GridSearchCV
**What this code does**: This section uses GridSearchCV to find the best hyperparameters for the Random Forest model by testing different combinations of `n_estimators` (number of trees) and `max_features` (number of features considered at each split). It also evaluates feature importance.

**How it relates to the goal**: Tuning the Random Forest model improves its accuracy, ensuring better house price predictions. Feature importance helps identify which variables (e.g., median income) are most critical for predictions.

```python
# Define hyperparameter grid for Random Forest
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},  # Test different numbers of trees and features
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}  # Test without bootstrap sampling
]

# Create GridSearchCV object
forest_reg = RandomForestRegressor()  # Initialize random forest model
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,  # Initialize grid search with 5-fold cross-validation
                           scoring='neg_mean_squared_error', return_train_score=True)  # Use MSE as metric
grid_search.fit(housing_prepared, housing_labels)  # Run grid search to find best parameters

# Display best hyperparameters and model
print(grid_search.best_params_)  # Show the best combination of hyperparameters
print(grid_search.best_estimator_)  # Show the best Random Forest model

# Display evaluation scores for each parameter combination
cvres = grid_search.cv_results_  # Get cross-validation results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # Loop through scores and parameters
    print(np.sqrt(-mean_score), params)  # Convert negative MSE to RMSE and show parameters

# Analyze feature importance
feature_importances = grid_search.best_estimator_.feature_importances_  # Get importance scores for each feature
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]  # List of new features
cat_encoder = full_pipeline.named_transformers_["cat"]  # Get the one-hot encoder from pipeline
cat_one_hot_attribs = list(cat_encoder.categories_[0])  # Get categorical feature names
attributes = num_attribs + extra_attribs + cat_one_hot_attribs  # Combine all feature names
sorted(zip(feature_importances, attributes), reverse=True)  # Sort features by importance
```

---

### 9. Final Model Evaluation
**What this code does**: This section uses the best Random Forest model from GridSearchCV to make predictions on the test set and computes the RMSE. It also calculates a 95% confidence interval for the error to quantify prediction uncertainty.

**How it relates to the goal**: The final evaluation confirms the model’s performance on unseen data, ensuring it can reliably predict house prices. The confidence interval provides insight into the model’s reliability.

```python
# Use the best model from grid search
final_model = grid_search.best_estimator_  # Select the best Random Forest model

# Prepare test set
X_test = strat_test_set.drop("median_house_value", axis=1)  # Separate test set features
y_test = strat_test_set["median_house_value"].copy()  # Store test set target values
X_test_prepared = full_pipeline.transform(X_test)  # Prepare test set using pipeline

# Make predictions with final model
final_predictions = final_model.predict(X_test_prepared)  # Predict house prices on test set

# Evaluate final model
final_mse = mean_squared_error(y_test, final_predictions)  # Compute mean squared error
final_rmse = np.sqrt(final_mse)  # Compute root mean squared error
print("Final Random Forest RMSE (test):", final_rmse)  # Display test set performance

# Calculate 95% confidence interval for the error
confidence = 0.95  # Set confidence level
squared_errors = (final_predictions - y_test) ** 2  # Compute squared errors
conf_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors)-1,  # Compute confidence interval
                                         loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
print("95% Confidence Interval for RMSE:", conf_interval)  # Display confidence interval
```

---

## Conclusion
This project builds a house price predictor using the California Housing dataset. The steps include:
1. **Data Preparation**: Downloading, loading, and splitting the data into training and test sets.
2. **Exploration**: Visualizing distributions and correlations to understand key features.
3. **Feature Engineering**: Creating new features like rooms per household to improve predictions.
4. **Preprocessing**: Handling missing values, encoding categorical variables, and scaling features.
5. **Model Training**: Testing Linear Regression, Decision Tree, and Random Forest models.
6. **Optimization**: Tuning the Random Forest model with GridSearchCV for best performance.
7. **Evaluation**: Measuring the final model’s accuracy with RMSE and a confidence interval.

The Random Forest model typically performs best due to its ability to capture complex patterns. The final RMSE and confidence interval provide a reliable estimate of prediction accuracy, ensuring the model is ready for real-world use.

## How to Use the Model
To predict house prices for new data:
1. Prepare the data in the same format as the training set (same features).
2. Apply the `full_pipeline` to preprocess the data.
3. Use `final_model.predict()` to get house price predictions.

This document and code ensure the knowledge is preserved and can be reused or extended for future house price prediction tasks.