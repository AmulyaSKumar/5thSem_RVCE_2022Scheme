import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression model
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

# Function to take user input for the dataset
def get_user_input():
    num_samples = int(input("Enter the number of samples: "))
    num_features = int(input("Enter the number of features: "))

    print("\nEnter the feature values row by row (space-separated):")
    X = []
    for _ in range(num_samples):
        row = list(map(float, input().split()))
        X.append(row)

    print("\nEnter the corresponding labels (0 or 1):")
    y = list(map(int, input().split()))

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
