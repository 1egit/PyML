import numpy as np

class KMeans:

    def __init__(self, data, K):

        self.data = data
        self.K = K

    def train(self, max_iter):

        centroids = self.init_centroids(self.data, self.K)
        ids = np.empty((self.data.shape[0], 1))

        for _ in range(max_iter):
            ids = self.find_ids(self.data, centroids)
            centroids = self.new_centroid(self.data, ids, self.K)

        return centroids, ids


    def init_centroids(self, data, num_centroids):
        # pick random K data in dataset as centroids initiated
        ind = np.random.choice(data.shape[0], num_centroids, replace=False)
        centroids = data[ind, :]
        print('centroid init ' + str(centroids))
        return centroids

    def find_ids(self, data, centroids):

        num_data = data.shape[0]
        num_centroids = centroids.shape[0]
#        ids = np.zeros((num_data,1))
#        this will make ids a 2d-array, lead to 'flatten()' being use in new_centroid()
        ids = np.zeros((num_data))

        for i in range(num_data):
#             print('find id for point ' + str(i))
            distances = np.zeros((num_centroids, 1))
            min_distance = float('inf')
            min_id = 0
            for j in range(num_centroids):
                diff = data[i, :] - centroids[j, :]
                distance = np.sum(diff ** 2)
                if distance < min_distance:
                    min_distance = distance
                    min_id = j
#                     print('min_distance for point '+ str(i) + ' ' + str(min_distance))
#                     print('min_id for point '+ str(i) + ' ' + str(min_id))
                    ids[i] = j
        return ids

    def new_centroid(self, data, ids, K):

        centroids = np.zeros((K, data.shape[1]))

        for i in range(K):

            i_ids = ids == i
#             print(i_ids)
            # centroids[i] = np.mean(data[i_ids.flatten(), :], axis = 0)
            # i_ids is a 2d array of True and False, which need to be flatten
            centroids[i] = np.mean(data[i_ids, :], axis = 0)
        print(centroids)

        return centroids

import numpy as np
import pandas as pd
data = pd.read_csv('data/iris.csv')
num_data = data.shape[0]
#print(data)
x_train = data.values.reshape((num_data,5))[:,:-1]
print(x_train)
print(type(x_train))
# ind = np.array([1,2,3])

kmeans = KMeans(x_train,3)
kmeans.train(1000)
