from sklearn import datasets, cluster

# Load dataset
X = datasets.load_iris().data

# Specify the parameters for clustering.
db = cluster.DBSCAN(eps=0.5, min_samples=5)
db.fit(X)