K-Means Clustering Implementation

## Installation
Ensure you have Python installed on your system. Then, install the required packages:

```bash
pip install  matplotlib scikit-learn
```

---

## Code
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)

def kmeans(X, k):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(100):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    return centroids, labels

# Apply custom k-means clustering
k = 3
centroids, labels = kmeans(X, k)

# Define colors for each cluster
colors = ['r', 'g', 'b']

# Plot the original data points with different colors for each cluster
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i+1}')

# Plot the final cluster centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', label='Centroids')

plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
```

---

## Output
When the code is executed, you will see a scatter plot showing the data points grouped into 3 clusters, with different colors representing each cluster. The final centroids are marked with black "X" symbols.

Example of the expected output:

- Red, green, and blue points represent the 3 clusters.
- Black "X" marks the centroids of the clusters.

---

## Advantages of K-Means
1. **Simplicity**: Easy to understand and implement.
2. **Efficiency**: Works well with large datasets.
3. **Flexibility**: Can adapt to different types of data.
4. **Speed**: Computationally efficient compared to other clustering algorithms.

---

## Disadvantages of K-Means
1. **Initialization Dependency**: Results depend on the initial placement of centroids.
2. **Fixed Number of Clusters**: Requires predefining the number of clusters (k).
3. **Sensitivity to Outliers**: Outliers can significantly affect cluster assignments.
4. **Shape Limitations**: Assumes clusters are spherical and equally sized.

---

## Viva Questions
1. **What is K-Means clustering?**
   - K-Means is an unsupervised machine learning algorithm that groups data into k clusters by minimizing the variance within each cluster.

2. **What is the role of centroids in K-Means?**
   - Centroids represent the center of each cluster and are updated iteratively to minimize the distance from data points.

3. **What distance metric is used in K-Means?**
   - The Euclidean distance is commonly used.

4. **Why is the number of clusters (k) important?**
   - The choice of k affects the accuracy and interpretability of the clustering results.

5. **What are some real-world applications of K-Means?**
   - Market Segmentation: Businesses use K-Means to segment their customer base into different groups based on purchasing behavior, demographics, and other attributes. This helps in targeted marketing, personalized offers, and improving customer retention.
   - Image Compression: K-Means is used in image processing to reduce the size of an image while preserving its quality. By clustering similar colors into a few representative ones, K-Means compresses the image with minimal loss of detail.
   - Document Clustering: In natural language processing (NLP), K-Means can be applied to group documents with similar themes or topics. This is useful for organizing large collections of text data, such as news articles, research papers, or customer reviews.
   - Anomaly Detection: K-Means can detect outliers or anomalies in data by identifying data points that do not fit into any cluster. This is helpful in fraud detection, network security, or quality control in manufacturing processes.

6. **How does K-Means handle outliers?**
   - Poorly, as outliers can skew the centroids and affect the cluster assignments.


---

