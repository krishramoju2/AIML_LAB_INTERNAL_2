

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

kmeans = KMeans(n_clusters=3)
agglo = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN()

print("kmeans")
kmeans_labels = kmeans.fit_predict(X)
print("Accuracy:", accuracy_score(y, kmeans_labels))

print("agglo")
agglo_labels = agglo.fit_predict(X)
print("Accuracy:", accuracy_score(y, agglo_labels))

print("dbscan")
dbscan_labels = dbscan.fit_predict(X)
print("Accuracy:", accuracy_score(y, dbscan_labels))
