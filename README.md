# Customer-Churn-Prediction-in-a-Telecom-Company
Customer Churn Prediction in a Case Study Telecom Company

![icons8-team-r-enAOPw8Rs-unsplash](https://github.com/Waltberry/Customer-Churn-Prediction-in-a-Telecom-Company/assets/63509339/ff8462e5-6e1a-4511-9f13-e1309dbed1cb)
Credits to: <a href="https://unsplash.com/@icons8?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Icons8 Team</a> on <a href="https://unsplash.com/photos/r-enAOPw8Rs?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  

# Case Study: Customer Churn Prediction in Telecom

## Data Cleaning and Preprocessing

### Import/Loading dataset
In this project, you and I will explore the task of predicting customer churn in a fictional telecom industry. Customer churn, often referred to as customer attrition, is when customers discontinue using a company's services. Accurately predicting churn is crucial for businesses as it allows them to take proactive measures to retain customers.

### About the Author
I am Onyero Walter Ofuzim, a professional Data Scientist with a passion for leveraging data-driven insights to solve complex business problems. With a strong background in data analysis, machine learning, and predictive modelling, I specialize in helping organizations make informed decisions based on data. This project showcases my expertise in data preprocessing, feature engineering, model building, and deployment.

If you have any questions or inquiries, feel free to reach out to me at [onyero.ofuzim@eng.uniben.edu](mailto:onyero.ofuzim@eng.uniben.edu).

### Import Necessary Libraries
To begin our analysis, we need to import various libraries for data processing, data cleaning, model building, and deployment. Below are the libraries we will be using in this project:

```python
# Data processing/ data cleaning libraries
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
```

### Importing the Dataset
We will start by loading the telecom customer churn dataset, which is stored in a CSV file. The dataset will serve as the foundation for our analysis.

```python
# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Viewing the dataset
df.head()
```

### Handling Inconsistencies and Investigative Analysis
We perform an initial investigation into the dataset to identify any missing values and ensure that the data types are accurate. It's essential to have a clear understanding of the data's quality and structure before proceeding with further analysis.

```python
# Investigating if there are missing values
plt.figure(figsize=(6, 2))
plt.title('Telco Customer')
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# No missing values found

# Investigating on the data types of each column
print(df.info())

# Investigating on the columns to check if the data types are accurate
def check_dtype(columns=df.columns):
    for column in columns:
        unique_values = df[column].unique()
        data_types = df[column].dtypes
        print(f"Unique values in {column}: {unique_values}, dtype in {column}: {data_types}")

check_dtype()
```

### Data Transformation
We perform some data transformations, including converting the 'SeniorCitizen' column to categorical data, handling the 'TotalCharges' column's data type inconsistencies, and dealing with missing values.

```python
# Converting Churn to numerical for correlation analysis
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

# Converting 'SeniorCitizen' to categorical
df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})

# Checking non-numeric values in 'TotalCharges'
print(df[~df['TotalCharges'].str.isnumeric()]['TotalCharges'].unique())

# Convert 'TotalCharges' to numeric datatype, handling non-numeric values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Checking for NaN values in 'TotalCharges'
print(f"Number of NaN values in TotalCharges: {df['TotalCharges'].isna().sum()}")

# Dropping rows with NaN values in 'TotalCharges'
df.dropna(subset=['TotalCharges'], inplace=True)

# Rechecking for NaN values in 'TotalCharges'
print(f"Number of NaN values in TotalCharges: {df['TotalCharges'].isna().sum()}")

# Rechecking data types after transformations
check_dtype(['SeniorCitizen', 'TotalCharges', 'Churn'])  # All data types are now accurate
```

### Feature Engineering
Feature engineering involves creating new features or transforming existing ones to improve the predictive power of our models. In this project, we've added a 'MonthlyCharges_log' feature by taking the natural logarithm of the 'MonthlyCharges' column.

```python
# Feature Engineering: Adding 'MonthlyCharges_log'
df['MonthlyCharges_log'] = np.log1p(df['MonthlyCharges'])

# df['TotalMonetaryValue'] = df['tenure'] * df['MonthlyCharges']

df.head()
```

### Feature Selection (Dimensionality Reduction)
Feature selection helps us choose the most relevant features for our predictive models. We perform both correlation analysis and univariate feature selection to identify important features.

#### Correlation Analysis
We calculate the Pearson correlation coefficients between numerical features and the target variable 'Churn' to select relevant numerical features.

```python
# Correlation Analysis: Selecting numerical features
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
```

#### Univariate Feature Selection
For categorical features, we perform a chi-squared test to identify important features based on their p-values.

```python
# Univariate Feature Selection: Selecting categorical features
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
selected_c_features = [feature for feature, p in p_values.items() if p < significance_level

]

# Display the selected features
print("Selected Categorical Features:")
print(selected_c_features)
```

### Visual Exploratory Data Analysis (EDA)
Visual EDA helps us gain insights into the relationships between features and the target variable. We create pair plots, correlation heatmaps, chi-squared p-value plots, and box plots for selected features.

```python
# Pair plots for selected numerical features
sns.pairplot(data=df[selected_n_features])

# Correlation heatmap
correlations = df[selected_n_features].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Chi-squared p-value plots for selected categorical features
for column in selected_c_features:
    contingency_table = pd.crosstab(df[column], df['Churn'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f'{column} Chi-Square p-value: {p}')

# Stacked bar plots for selected categorical features
for column in selected_c_features:
    contingency_table = pd.crosstab(df[column], df['Churn'])
    contingency_table.plot(kind='bar', stacked=True)
    plt.title(f'Stacked Bar Plot of {column} vs. Churn')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Churn', labels=['0', '1'])
    plt.show()

# Box plots for selected numerical features
for var in selected_n_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Churn', y=var, data=df)
    plt.title(f'{var} vs. Churn')
    plt.show()
```

## Model Building

### List of Columns to Keep
We create a list of columns to keep, including both selected categorical and numerical features.

```python
# List of columns to keep
columns_to_keep = selected_c_features + selected_n_features

# Select and keep only the desired columns
df = df[columns_to_keep]
print(columns_to_keep)
```

### Splitting Data
Before building models, we split the data into training and testing sets. Features (X) and the target variable (y) are separated accordingly.

```python
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
```

### Training and Evaluating
We train multiple machine learning models and evaluate their performance using metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

```python
# Label Encoding for ordinal variables
label_encoder = LabelEncoder()
for feature in selected_c_features:
    X_train[feature] = label_encoder.fit_transform(X_train[feature])
    X_test[feature] = label_encoder.transform(X_test[feature])

# One-Hot Encoding for nominal variables
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
```

## Deployment

### Model Deployment
After evaluating the models, we select the best-performing model (in this case, Logistic Regression) and train it on the entire training dataset. We then save the trained model for future use.

```python
# Train the best model (Logistic Regression)
best_model = LogisticRegression()
best_model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'logistic_regression_model.joblib'
joblib.dump(best_model, model_filename)
```

### Model Loading and Prediction
To deploy the model, you can load it from the saved file and use it to make predictions.

```python
# Load the logistic regression model from the saved file
loaded_model = joblib.load(model_filename)

# Perform predictions using the loaded model
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
print(f

"ROC AUC: {roc_auc}")
```
