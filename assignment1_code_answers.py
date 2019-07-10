# Question 4 -


from collections import Counter

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

plt.style.use('ggplot')

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://raw.githubusercontent.com/m-braverman/ta_dm_course_data/master/x_y_terrorism_data.csv')
print("dfdfd")

df_std = StandardScaler().fit_transform(df)
# plt.scatter(df_std[:, 1], df_std[:, 0])
#
# clustering = DBSCAN(eps=3, min_samples=2).fit(df_std)
#
# # Black removed and is used for noise instead.
# core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
# core_samples_mask[clustering.core_sample_indices_] = True
# labels = clustering.labels_
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     xy = df_std[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)
#
#     xy = df_std[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)
#
# plt.title('Cluster is - ')
# plt.show()

ns = 4
nbrs = NearestNeighbors(n_neighbors=ns).fit(df_std)
distances, indices = nbrs.kneighbors(df_std)
distanceDec = sorted(distances[:,ns-1], reverse=True)

plt.plot(indices[:,0], distanceDec)

clustering = KMeans(n_clusters=21).fit(df_std)
clustering.