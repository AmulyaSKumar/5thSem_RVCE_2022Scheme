import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
y = iris.target  # Labels: 0 (setosa), 1 (versicolor), 2 (virginica)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# KNN implementation
def knn_predict(X_train, y_train, test_point, k=3):
    # Calculate distances from the test point to all training points
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((dist, y_train[i]))  # (distance, label)

    # Sort distances and pick the top k
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]

    # Count the occurrences of each class in the k nearest neighbors
    class_votes = {}
    for _, label in k_neighbors:
        class_votes[label] = class_votes.get(label, 0) + 1

    # Return the class with the most votes
    return max(class_votes, key=class_votes.get)

# Testing the KNN implementation
k = 3
y_pred = []
for test_point in X_test:
    prediction = knn_predict(X_train, y_train, test_point, k)
    y_pred.append(prediction)

# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"KNN Accuracy: {accuracy:.2f}")

# Example: Predicting a new sample
new_sample = np.array([5.0, 3.6, 1.4, 0.2])  # Example input
predicted_class = knn_predict(X_train, y_train, new_sample, k)
print(f"Predicted class for the sample: {iris.target_names[predicted_class]}")
