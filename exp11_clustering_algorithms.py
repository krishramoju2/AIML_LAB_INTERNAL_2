from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

blobs_data, _ = make_blobs(n_samples=300, centers=4, random_state=42)
moons_data, _ = make_moons(n_samples=300, random_state=42)

scaler = StandardScaler()
blobs_scaled = scaler.fit_transform(blobs_data)
moons_scaled = scaler.fit_transform(moons_data)

datasets = [('Blobs', blobs_scaled), ('Moons', moons_scaled)]
algos = [
    ('KMeans', KMeans(n_clusters=4, random_state=42)),
    ('Agglomerative', AgglomerativeClustering(n_clusters=4)),
    ('DBSCAN', DBSCAN(eps=0.5, min_samples=5))
]

for d_name, d_data in datasets:
    print(f"\nResults for {d_name}:")
    for a_name, model in algos:
        labels = model.fit_predict(d_data)
        n_clusters = len(set(labels))
        if n_clusters > 1:
            sil = silhouette_score(d_data, labels)
        else:
            sil = 0.0
        print(f"  {a_name:<15} -> Clusters: {n_clusters}, Silhouette: {sil:.3f}")
      
