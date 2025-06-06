import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

sil_score = silhouette_score(X, clusters)
print(f"Silhouette Score: {sil_score:.3f}")

colors = ['red', 'blue', 'green', 'purple', 'orange']
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.scatter(
        X[clusters == i]['Annual Income (k$)'],
        X[clusters == i]['Spending Score (1-100)'],
        s=100, c=colors[i], label=f'Cluster {i}'
    )
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    s=300, c='yellow', marker='X', label='Centroids'
)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments - KMeans Clustering')
plt.legend()
plt.grid(True)
plt.show()
