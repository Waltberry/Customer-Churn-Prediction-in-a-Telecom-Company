# -*- coding: utf-8 -*-
"""Customer_Churn_Prediction_in_Telecom.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rtZ-fXHgl-eF6uM3R_vQcj1Dri3TQUd0

# Case Study: Customer Churn Prediction in Telecom

## Data Cleaning and Preprocessing

### Import/Loading dataset
"""

# Import necessary libraries

# data processing/ data cleaning libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency

# Model Building libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Deployment libraries
import joblib

# Commented out IPython magic to ensure Python compatibility.
# Run this code if you using Google Colab
# First upload the dataset in your google drive
from google.colab import drive
drive.mount('/content/drive')
# %cd drive
# %cd My Drive
# %cd 'Customer Churn Dataset '/

# Load dataset
df =pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Viewing the dataset
df.head()

# Having an idea of the features involved
df.columns

# 21 features, 1 target feature

"""### Handling Inconsistency/ Investigative Analysis"""

# Investigating if they are missing values
plt.figure(figsize=(6,2))
plt.title('Telcom Customer')
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# No missing values found

# Investigating on the data types of each columns
print(df.info())

# investigating on the columns to check if the data types are accurate.
def check_dtype(columns=df.columns):
    for column in columns:
        unique_values = df[column].unique()
        data_types = df[column].dtypes
        print(f"Unique values in {column}:" + f" dtype in {column}:")
        print(unique_values, data_types)

check_dtype()

"""`Take-offs`

1. **SeniorCitizen:** This should be treated as categorical data with two categories:
   - `0`: Not a senior citizen
   - `1`: A senior citizen

2. **TotalCharges:** This should be considered as numerical data.

"""

# Converting Churn to numerical for Correlation analysis.
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

# Solving this issue: 1.
df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})

# Issue 2.
# Checking non numeric data
print(df[~df['TotalCharges'].str.isnumeric()]['TotalCharges'].unique())

# Convert to numeric datatype
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# errors='coerce' argument will convert non-numeric values to NaN.

# Check for NaN values

print(f"Number of NaN values in TotalCharges: {df['TotalCharges'].isna().sum()}")

# I decided to drop the NaN because it is 11 and not a lot compared to the data: 7043
df.dropna(subset=['TotalCharges'], inplace=True)
print(f"Number of NaN values in TotalCharges: {df['TotalCharges'].isna().sum()}")

# Investigating to make sure our processes worked well.
check_dtype(['SeniorCitizen', 'TotalCharges', 'Churn']) # All done.

"""### Feature Engineering"""

# Feature Engineering by adding features you think might improve the model
df['MonthlyCharges_log'] = np.log1p(df['MonthlyCharges'])


# df['TotalMonetaryValue'] = df['tenure'] * df['MonthlyCharges']

df.head()

"""### Feature Selection (Dimensionality Reduction)"""

# Correlation Analysis

# Select only numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64'])

# Calculate Pearson correlation coefficients with the target variable
correlation_matrix = numerical_features.corr()
correlation_with_target = correlation_matrix['Churn'].abs().sort_values(ascending=False)

# Set a correlation threshold
correlation_threshold = 0.1

# Select features with correlation above the threshold
selected_n_features = correlation_with_target[correlation_with_target > correlation_threshold].index.tolist()

# Display the selected features
print("Selected numerical Features:")
print(selected_n_features)

# Univariate Feature Selection
categorical_features = df.select_dtypes(include=['object', 'category'])

# Create a contingency table for each categorical feature vs. the target variable
contingency_tables = {}
for feature in categorical_features.columns:
    contingency_table = pd.crosstab(df[feature], df['Churn'])
    contingency_tables[feature] = contingency_table

# Calculate the chi-squared statistic and p-value for each feature
chi2_values = {}
p_values = {}
for feature, table in contingency_tables.items():
    chi2, p, _, _ = stats.chi2_contingency(table)
    chi2_values[feature] = chi2
    p_values[feature] = p

# Set a significance level (you can adjust this)
significance_level = 0.05

# Select features with p-values below the significance level
selected_c_features = [feature for feature, p in p_values.items() if p < significance_level]

# Display the selected features
print("Selected Categorical Features:")
print(selected_c_features)

"""### Visual Exploratory Data Analysis (EDA)"""

sns.pairplot(data=df[selected_n_features])

# summary_stats = donor_raw_data[['TARGET_B', 'TARGET_D', 'RECENT_RESPONSE_COUNT', 'RECENT_CARD_RESPONSE_COUNT', 'RESPONSE_RATE', 'RECENT_RESPONSE_PROP', 'FILE_CARD_GIFT']].describe()
correlations = df[selected_n_features].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

for column in selected_c_features:
    contingency_table = pd.crosstab(df[column], df['Churn'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f'{column} Chi-Square p-value: {p}')

for column in selected_c_features:
    contingency_table = pd.crosstab(df[column], df['Churn'])
    contingency_table.plot(kind='bar', stacked=True)
    plt.title(f'Stacked Bar Plot of {column} vs. Churn')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Churn', labels=['0', '1'])
    plt.show()

for var in selected_n_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Churn', y=var, data=df)
    plt.title(f'{var} vs. Churn')
    plt.show()

"""## Model Building

"""

# List of columns to keep
columns_to_keep = selected_c_features+selected_n_features


# Select and keep only the desired columns
df = df[columns_to_keep]

print(columns_to_keep)

"""### Spliting Data"""

# Split the data into features (X) and target variable (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']               # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:")
print(X_train.shape)  # Print the shape of the training features
print(y_train.shape)  # Print the shape of the training target variable

print("\nTesting data shape:")
print(X_test.shape)   # Print the shape of the testing features
print(y_test.shape)   # Print the shape of the testing target variable

"""### Training and Evaluating"""

# Use Label Encoding for ordinal variables
label_encoder = LabelEncoder()
for feature in selected_c_features:
    X_train[feature] = label_encoder.fit_transform(X_train[feature])
    X_test[feature] = label_encoder.transform(X_test[feature])

# Use One-Hot Encoding for nominal variables
X_train = pd.get_dummies(X_train, columns=selected_c_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=selected_c_features, drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
X_train[['tenure', 'MonthlyCharges_log', 'TotalCharges']] = scaler.fit_transform(X_train[['tenure', 'MonthlyCharges_log', 'TotalCharges']])
X_test[['tenure', 'MonthlyCharges_log', 'TotalCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges_log', 'TotalCharges']])

# Model training and evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(),
    "LightGBM": LGBMClassifier()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1-Score: {f1_score(y_test, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
    print("\n")

# Initialize variables to keep track of the best model and its accuracy
best_model_name = ""
best_accuracy = 0

# Loop through each model and perform cross-validation
for model_name, model in models.items():
    # Perform cross-validation (you can choose the number of folds as needed)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Calculate the mean accuracy from cross-validation
    mean_accuracy = scores.mean()

    # Check if this model has the highest accuracy so far
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_model_name = model_name

# Print the best model and its accuracy
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy}")

"""## Deployment"""

# Train the best model (Logistic Regression)
best_model = LogisticRegression()
best_model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'logistic_regression_model.joblib'
joblib.dump(best_model, model_filename)

# Load the logistic regression model from the saved file
loaded_model = joblib.load(model_filename)

# # Perform predictions using the loaded model
predictions = loaded_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Calculate precision
precision = precision_score(y_test, predictions)
print(f"Precision: {precision}")

# Calculate recall
recall = recall_score(y_test, predictions)
print(f"Recall: {recall}")

# Calculate F1-score
f1 = f1_score(y_test, predictions)
print(f"F1-Score: {f1}")

# Calculate ROC AUC score (if applicable for your task)
roc_auc = roc_auc_score(y_test, predictions)
print(f"ROC AUC: {roc_auc}")