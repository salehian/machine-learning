from scipy.cluster.hierarchy import dendrogram, ward, single
from sklearn import datasets
import matplotlib.pyplot as plt

# Load dataset
X = datasets.load_iris().data[:10]

# Perform clustering
linkage_matrix = ward(X)

# Plot dendogram
dendrogram(linkage_matrix)
plt.show()