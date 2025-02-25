## Logistic Regression Implementation

## Introduction
Logistic Regression is a fundamental classification algorithm used for binary classification problems. It estimates the probability that a given input belongs to a particular category. 

---

## Explanation of the Code

### Manual Implementation
1. **Data Loading**: The `load_iris()` function from Scikit-Learn is used to load the Iris dataset.
2. **Data Preprocessing**:
   - Filtering only Setosa and Versicolor classes (binary classification).
   - Standardizing the features using `StandardScaler`.
3. **Sigmoid Function**: Computes the probability of a data point belonging to a certain class.
4. **Logistic Regression Function**:
   - Initializes weights to zeros.
   - Uses gradient descent to update weights iteratively.
   - Computes the gradient based on the difference between predicted and actual values.
5. **Prediction Function**:
   - Transforms input features.
   - Applies the sigmoid function and makes a prediction based on probability.
6. **User Input Handling**:
   - Takes input features from the user.
   - Standardizes the input and makes predictions.
---

## Implementation Without Built-in Functions (Manual Approach)

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_regression(X, y, num_iterations=200, learning_rate=0.001):
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term
    weights = np.zeros(X.shape[1])  # Initialize weights
    for _ in range(num_iterations):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.shape[0]
        weights -= learning_rate * gradient
    return weights

def predict(input_data, weights, scaler):
    input_data_std = scaler.transform(input_data)
    input_data_std = np.hstack([np.ones((input_data_std.shape[0], 1)), input_data_std])  
    prediction = sigmoid(np.dot(input_data_std, weights)) > 0.5  
    return "Setosa" if prediction[0] == 0 else "Versicolor"

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Filter to include only Setosa and Versicolor
mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]

# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Train logistic regression model
weights = logistic_regression(X_std, y)

# User input for prediction
sepal_len = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_len = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

user_input = np.array([[sepal_len, sepal_width, petal_len, petal_width]])
result = predict(user_input, weights, scaler)
print(f"Prediction: {result}")
```

### Sample Output:
```
Enter sepal length: 5.1
Enter sepal width: 3.5
Enter petal length: 1.4
Enter petal width: 0.2
Prediction: Setosa
```

---

## Implementation Using Built-in Function (Scikit-Learn)

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Filter to include only Setosa and Versicolor
mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]

# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_std, y)

# User input for prediction
sepal_len = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_len = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

user_input = np.array([[sepal_len, sepal_width, petal_len, petal_width]])
user_input_std = scaler.transform(user_input)

prediction = model.predict(user_input_std)[0]
result = "Setosa" if prediction == 0 else "Versicolor"

print(f"Prediction: {result}")
```

### Sample Output:
```
Enter sepal length: 5.1
Enter sepal width: 3.5
Enter petal length: 1.4
Enter petal width: 0.2
Prediction: Setosa
```

---

## Advantages & Disadvantages

**Advantages:**
1. Simple and Interpretable: Easy to implement and interpret results.
2. Efficient for Binary Classification: Works well for problems with two possible outcomes.
3. Fast Training: Compared to complex models, logistic regression trains quickly.
4. Handles Linearly Separable Data Well: Performs well when classes can be separated by a linear boundary.
5. Probabilistic Output: Provides probability scores instead of just classifications.
6. Less Prone to Overfitting: With proper regularization (L1/L2), it generalizes well.
7. Feature Importance: Coefficients indicate the significance of features in classification.

**Disadvantages:**
1. Limited to Linear Boundaries: Does not perform well when decision boundaries are highly non-linear.
2. Sensitive to Outliers: Can be affected significantly by extreme data points.
3. Assumes No Multicollinearity: Assumes features are independent; correlated features can distort predictions.
4. Not Suitable for Large Feature Sets: Can struggle with datasets having a high number of features without proper dimensionality reduction.
5. Requires Feature Scaling: Performance depends on proper feature normalization or standardization.
6. Binary Classification Limitation: Needs modifications like One-vs-Rest (OvR) or Softmax for multi-class problems.

## Applications of Logistic Regression
1. Medical Diagnosis: Used to predict diseases based on symptoms.
2. Customer Churn Prediction: Predicts if a customer is likely to stop using a service.
3. Spam Detection: Classifies emails as spam or non-spam.
4. Credit Scoring: Determines whether a loan applicant is likely to default.
5. Marketing: Predicts whether a customer will buy a product based on past behavior.

---

## Few Questions 
1. **What is logistic regression, and how does it differ from linear regression?**  
   Logistic regression is used for classification, while linear regression is used for predicting continuous values. Logistic regression applies the sigmoid function to convert outputs into probabilities.

2. **Explain the sigmoid function and its role in logistic regression.**  
   The sigmoid function maps input values to a range between 0 and 1, making it useful for probability estimation in binary classification.

3. **How do you interpret the weights in logistic regression?**  
   The weights determine the importance of each feature. A higher weight indicates a stronger influence on the prediction.

4. **Why do we standardize features before applying logistic regression?**  
   Standardization improves model convergence and performance by ensuring all features have the same scale.

5. **What are the advantages of using built-in functions for logistic regression?**  
   Built-in functions are optimized, well-tested, and easier to use, reducing the need for manual implementation.

6. **What are some real-world applications of logistic regression?**  
   It is used in medical diagnosis, fraud detection, spam filtering, and customer churn prediction.

7. **What is the difference between binary and multi-class logistic regression?**  
   Binary logistic regression deals with two classes, while multi-class logistic regression handles more than two classes using techniques like one-vs-rest or softmax.

8. **How does logistic regression handle overfitting?**  
   Regularization techniques like L1 (Lasso) and L2 (Ridge) help prevent overfitting.

9. **What is the role of the learning rate in gradient descent?**  
   The learning rate controls the step size for weight updates during optimization.

10. **How can logistic regression be used for multi-class classification?**  
    It can be extended using methods like one-vs-rest (OvR) and softmax regression.



