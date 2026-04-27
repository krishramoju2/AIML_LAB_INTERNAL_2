print("="*50)
print("Experiment 11: Clustering Algorithms")
print("="*50)

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

kmeans = KMeans(n_clusters=3)
agglo = AgglomerativeClustering(n_clusters=3)

kmeans_labels = kmeans.fit_predict(X)
print("\nKMeans:")
print("Accuracy:", accuracy_score(y, kmeans_labels), 4)


agglo_labels = agglo.fit_predict(X)
print("\nAgglomerative:")
print("Accuracy:", accuracy_score(y, agglo_labels))
