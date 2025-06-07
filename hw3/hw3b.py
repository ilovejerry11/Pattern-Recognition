import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 載入資料並標準化
iris = load_iris()
X = iris.data
y_true = iris.target
X_std = StandardScaler().fit_transform(X)

# --- KMeans ---
def kmeans(X, k, max_iter=100):
    np.random.seed(42)
    idx = np.random.choice(len(X), k, replace=False)
    centers = X[idx]
    for _ in range(max_iter):
        # 分配每個點到最近的中心
        labels = np.argmin(np.linalg.norm(X[:, None] - centers[None, :], axis=2), axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels

# --- Agglomerative Clustering (Single Linkage) ---
def agglomerative_single_linkage(X, n_clusters):
    clusters = [[i] for i in range(len(X))]
    distances = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    while len(clusters) > n_clusters:
        # 找到最近的兩個群
        min_dist = np.inf
        to_merge = (None, None)
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                d = np.min([distances[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                if d < min_dist:
                    min_dist = d
                    to_merge = (i, j)
        # 合併
        i, j = to_merge
        clusters[i] += clusters[j]
        del clusters[j]
    # 產生 labels
    labels = np.zeros(len(X), dtype=int)
    for idx, cluster in enumerate(clusters):
        for i in cluster:
            labels[i] = idx
    return labels

def plot_clusters(X, labels, title):
    X_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(4, 3))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()

def print_scores(X, labels, name):
    # avoid calculating metrics if all points are in one cluster or noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        print(f"{name}:  can not calculate evaluation metrics (n_clusters={n_clusters})")
        return
    print(f"{name} Silhouette: {silhouette_score(X, labels):.3f}")
    # print(f"{name} Calinski-Harabasz: {calinski_harabasz_score(X, labels):.3f}")
    # print(f"{name} Davies-Bouldin: {davies_bouldin_score(X, labels):.3f}")
    print(f"{name} ARI: {adjusted_rand_score(y_true, labels):.3f}")
    # print(f"{name} NMI: {normalized_mutual_info_score(y_true, labels):.3f}")
    print("-" * 40)

# 執行 KMeans
kmeans_labels = kmeans(X_std, 3)
print_scores(X_std, kmeans_labels, "KMeans")
plot_clusters(X_std, kmeans_labels, "KMeans")

# 執行 Agglomerative Clustering
agg_labels = agglomerative_single_linkage(X_std, 3)
print_scores(X_std, agg_labels, "Agglomerative Single Linkage")
plot_clusters(X_std, agg_labels, "Agglomerative Single Linkage")

plt.show()