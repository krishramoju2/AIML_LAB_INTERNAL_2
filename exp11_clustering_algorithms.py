

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

kmeans = KMeans(n_clusters=3)
agglo = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN()

kmeans_labels = kmeans.fit_predict(X)
print("Accuracy:", accuracy_score(y, kmeans_labels))

agglo_labels = agglo.fit_predict(X)
print("Accuracy:", accuracy_score(y, agglo_labels))

dbscan_labels = dbscan.fit_predict(X)
print("Clusters found:", dbscan_labels)
