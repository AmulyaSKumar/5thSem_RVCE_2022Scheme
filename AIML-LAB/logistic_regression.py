from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_regression(X, y, num_iterations=200, learning_rate=0.001):
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term
    weights = np.zeros(X.shape[1])  # Initialize weights
    for _ in range(num_iterations):
        z = np.dot(X, weights)  # z = X * weights
        h = sigmoid(z)  # Sigmoid of z
        gradient = np.dot(X.T, (h - y)) / y.shape[0]  # Gradient of loss
        weights -= learning_rate * gradient  # Update weights
    return weights

def predict(input_data, weights, scaler):
    input_data_std = scaler.transform(input_data) 
    input_data_std = np.hstack([np.ones((input_data_std.shape[0], 1)), input_data_std])  
    prediction = sigmoid(np.dot(input_data_std, weights)) > 0.5  
    return "Setosa" if prediction[0] == 0 else "Versicolor"

# Load Iris dataset
iris = load_iris()
X = iris.data  # All features
y = iris.target  # All labels

# Filter to only include Setosa (label 0) and Versicolor (label 1)
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

# Standardize features using the entire dataset
sc = StandardScaler()
X_std = sc.fit_transform(X)

# Perform logistic regression on the full dataset
weights = logistic_regression(X_std, y)

# User input for prediction
sepal_len = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_len = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# Prepare user input
user_input = np.array([[sepal_len, sepal_width, petal_len, petal_width]])

# Use the predict function
result = predict(user_input, weights, sc)

<<<<<<< HEAD
    return np.array(X), np.array(y)
f
if __name__ == "__main__":
    # User inputs the dataset
    X, y = get_user_input()

    # Create and train the model
    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X, y)

    # Take user input for prediction
    print("\nEnter a new data point for prediction (space-separated values):")
    new_data = np.array(list(map(float, input().split()))).reshape(1, -1)

    # Predict and output the result
    prediction = model.predict(new_data)
    print(f"Prediction for the input data point: {prediction[0]}")
=======
print(f"Prediction: {result}")
>>>>>>> fe291e784d5ef6b07bee60f6261a2a3b23c5cfa2
