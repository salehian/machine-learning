from sklearn import datasets, cluster

# Load dataset
X = datasets.load_iris().data[:10]

# Specify parameters for the clustering. 'ward', 'complete', 'average' linkage
clusterer = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward')

# Predictions
labels = clusterer.fit_predict(X)
print(labels)