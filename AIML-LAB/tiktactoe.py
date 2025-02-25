import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=1, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot clusters
colors = ['r', 'g', 'b']

for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', label='Centroids')
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Predict cluster for user input
print("\nEnter feature values for clustering:")
user_input = []
feature_names = iris.feature_names
for feature in feature_names:
    value = float(input(f"{feature}: "))
    user_input.append(value)

user_pred = kmeans.predict([np.array(user_input)])
print("Predicted cluster:", user_pred[0])
