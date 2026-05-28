# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv('./bank-full.csv', sep=';')

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(data.head())

# -------------------------------
# Data Preprocessing
# -------------------------------

# Convert categorical columns into numerical values
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# -------------------------------
# Feature Selection
# -------------------------------

# Features
X = data.drop('y', axis=1)

# Target variable
y = data['y']

# -------------------------------
# Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Build Decision Tree Model
# -------------------------------

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# -------------------------------
# Model Prediction
# -------------------------------

predictions = model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# -------------------------------
# Visualization
# -------------------------------

# Count plot for target variable
target_counts = data['y'].value_counts()

plt.figure(figsize=(6,4))
target_counts.plot(kind='bar')

plt.title('Customer Subscription Distribution')
plt.xlabel('Subscription')
plt.ylabel('Count')

plt.show()
