from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
data = pd.read_csv("heart.csv")

# Remove rows with missing values
data_cleaned = data.dropna()

# Split data into features (X) and target variable (y)
X = data_cleaned.drop("target", axis=1)  # All columns except target
y = data_cleaned["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
# model = LogisticRegression()
model = LogisticRegression(solver='liblinear')

model.fit(X_train, y_train)

# Make predictions on test set
predictions = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# Use the trained model for new predictions
new_patient_data = {
    "age": 90,
    "sex": 1,
    "cp": 1,
    "trestbps": 125,
    "chol": 320,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1,
    "slope": 2,
    "ca": 2,
    "thal": 0
}

data_predict = pd.DataFrame([new_patient_data])
 # Replace with new data
new_prediction = model.predict_proba(pd.DataFrame(data_predict))[:, 1][0]
print(f"Predicted probability of heart disease: {new_prediction:.2f}")
