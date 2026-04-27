

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

kmeans = KMeans(n_clusters=3)
agglo = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN()

kmeans_labels = kmeans.fit_predict(X)
print("\nKMeans:")
print("Accuracy:", round(accuracy_score(y, kmeans_labels), 4))

agglo_labels = agglo.fit_predict(X)
print("\nAgglomerative:")
print("Accuracy:", round(accuracy_score(y, agglo_labels), 4))

dbscan_labels = dbscan.fit_predict(X)
print("\nDBSCAN:")
print("Clusters found:", len(set(dbscan_labels)))
