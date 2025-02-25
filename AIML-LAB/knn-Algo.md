# K-Nearest Neighbors (KNN) Implementation for Iris Dataset


KNN is a non-parametric, instance-based learning algorithm. It does not assume any specific form for the underlying data distribution. Instead, it relies on the distances between data points to make predictions. The algorithm is simple yet effective, particularly for smaller datasets.
Formula for Euclidean Distance:
\[
d(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
\]
- **Applications**: Pattern recognition, recommender systems, medical diagnosis, and more.

## Code (Using test set from iris)
```python
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

```
## Code (Using test set from user inputed values)
```python
import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X_train = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
y_train = iris.target  # Labels: 0 (setosa), 1 (versicolor), 2 (virginica)

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
print("\nEnter the features for the new Iris flower sample:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

# Create a numpy array from user input
new_sample = np.array([sepal_length, sepal_width, petal_length, petal_width])

# Predict the class for the new sample
predicted_class = knn_predict(X_train, y_train, new_sample, k)
print(f"\nPredicted class for the sample: {iris.target_names[predicted_class]}")
```

---

## Example Input and Output

```
Enter the features for the new Iris flower sample:
Sepal length (cm): 5.1
Sepal width (cm): 3.5
Petal length (cm): 1.4
Petal width (cm): 0.2
Predicted class for the sample: setosa
```

### Advantages:
1. **Simplicity**: Easy to implement and understand.
2. **Flexibility**: Can be used for classification and regression tasks.
3. **No Training**: KNN is a lazy learner; it does not require training time.

### Disadvantages:
1. **Computational Cost**: High computation cost during testing as it calculates distances for all training points.
2. **Storage Requirements**: Requires storing the entire dataset.
3. **Sensitive to Noise**: Outliers can significantly affect predictions.
4. **Curse of Dimensionality**: Performance degrades with high-dimensional data.

---

## Viva Questions

1. **What is the KNN algorithm?**
   - KNN is a supervised learning algorithm that classifies a sample based on the majority class of its nearest neighbors.

2. **What are the main parameters of the KNN algorithm?**
   - The number of neighbors (k) and the distance metric (e.g., Euclidean distance).

3. **How does the value of `k` affect the performance of KNN?**
   - A small `k` can lead to overfitting, while a large `k` can lead to underfitting. The choice of `k` balances bias and variance.

4. **Why is KNN called a lazy learner?**
   - KNN does not build a model during training; it defers computation until prediction.

5. **What are some common distance metrics used in KNN?**
   - Euclidean distance, Manhattan distance, Minkowski distance, and Hamming distance.

6. **What is the impact of scaling features in KNN?**
   - Feature scaling ensures that all features contribute equally to the distance metric, preventing bias from large-scale features.

7. **Can KNN be used for regression tasks? If yes, how?**
   - Yes, KNN regression predicts the value by averaging the target values of the k-nearest neighbors.

8. **What are the limitations of KNN?**
   - High computation and memory requirements, sensitivity to noise, and challenges with high-dimensional data.

---
