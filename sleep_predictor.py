
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

print("First 5 rows of dataset:")
print(df.head())


print("\nDataset Info:")
print(df.info())


print("\nMissing values before cleaning:")
print(df.isnull().sum())


df = df.drop(['Sleep Disorder', 'Blood Pressure'], axis=1)


le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['Occupation'] = le.fit_transform(df['Occupation'])
df['BMI Category'] = le.fit_transform(df['BMI Category'])


X = df.drop('Quality of Sleep', axis=1)
y = df['Quality of Sleep']

print("\nFinal Feature Columns:")
print(X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

