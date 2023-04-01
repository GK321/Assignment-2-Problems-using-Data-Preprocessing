import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/tranghth-lux/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')

# Split into input and output variables
X = df.drop('left', axis=1)
y = df['left']

# Preprocess the data
numeric_features = ['satisfaction_level', 'last_evaluation', 'average_montly_hours']
numeric_transformer = StandardScaler()

categorical_features = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
score = clf.score(X_test, y_test)
print(f"Accuracy: {score}")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/tranghth-lux/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')

# Explore the data
print(df.head())
print(df.describe())
print(df.info())

# Visualize the data
sns.countplot(x='left', data=df)
plt.show()

sns.countplot(x='salary', hue='left', data=df)
plt.show()

sns.countplot(x='sales', hue='left', data=df)
plt.show()

sns.histplot(x='satisfaction_level', hue='left', data=df)
plt.show()

sns.histplot(x='last_evaluation', hue='left', data=df)
plt.show()

sns.histplot(x='average_montly_hours', hue='left', data=df)
plt.show()

# Feature engineering
df['satisfaction_evaluation_ratio'] = df['satisfaction_level'] / df['last_evaluation']
df['satisfaction_evaluation_ratio'].fillna(0, inplace=True)

# Split into input and output variables
X = df.drop('left', axis=1)
y = df['left']

# Preprocess the data
numeric_features = ['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'satisfaction_evaluation_ratio']
numeric_transformer = StandardScaler()

categorical_features = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
score = clf.score(X_test, y_test)
print(f"Accuracy: {score}")

# Train a random forest model
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
rfc = RandomForestClassifier(random_state=42)
clf_rfc = GridSearchCV(rfc, param_grid=param_grid, cv=5, n_jobs=-1)
clf_rfc.fit(X_train, y_train)

# Evaluate the model
score = clf_rfc.score(X_test, y_test)
print(f"Accuracy: {score}")

# Print classification report and confusion matrix
y_pred = clf_rfc.predict(X_test)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(clf_rfc, X_test, y_test)
plt.show()
