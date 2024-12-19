import numpy as np
from sklearn.cluster import KMeans

# create the data
data = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# create the model
# the number of clusters = k value
kmeans = KMeans(n_clusters=2, random_state=0)

# train the model
kmeans.fit(data)

# print the output
print(kmeans.labels_)

# print the centroids
print(kmeans.cluster_centers_)

