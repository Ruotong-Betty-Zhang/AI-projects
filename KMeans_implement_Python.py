import numpy as np
import matplotlib.pyplot as plt

def kmeans(data, k, max_iter=100):
    # randomly select k centroids
    # choose k data points randomly from the data
    # data.shape[0] is the number of data points
    centers = data[np.random.choice(data.shape[0], k, replace=False)]

    # create an array to store the labels of the data points
    labels = np.zeros(data.shape[0])

    for i in range(max_iter):
        # calculate the distance between each data point and the centroids
        distances = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))

        # assign each data point to the closest centroid
        new_labels = np.argmin(distances, axis=0)

        for j in range(k):
            # update the centroids
            centers[j] = data[new_labels == j].mean(axis=0)

        # if the labels do not change, stop the iteration
        if np.all(new_labels == labels):
            break
        else:
            labels = new_labels
    return labels, centers

# # create the data
# data = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
#
# # train the model
# labels, centroids = kmeans(data, 2)
#
# # print the output
# print(labels)
#
# # print the centroids
# print(centroids)

# use another dataset
dataset = np.vstack((np.random.randn(100, 2) * 0.75 + np.array([1, 0]),
               np.random.randn(100, 2) * 0.25 + np.array([-0.5, 0.5]),
               np.random.randn(100, 2) * 0.5 + np.array([-0.5, -0.5])))

# run the model
labels, centroids = kmeans(dataset, 3)

# visualize the output
plt.scatter(dataset[:, 0], dataset[:, 1], c=labels)

# plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x', s=200, linewidths=3)

plt.show()













