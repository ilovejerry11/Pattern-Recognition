from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, MeanShift
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 載入資料
iris = load_iris()
X = iris.data
y_true = iris.target

# 標準化
X_std = StandardScaler().fit_transform(X)

def print_scores(X, labels, name):
    # 避免所有點都被分到同一群或雜訊
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

def plot_clusters(X, labels, title):
    # 使用PCA將資料降到2維再畫圖
    X_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(4, 3))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()

# 0. original data
plot_clusters(X_std, y_true, "Ground Truth")

# 1. KMeans
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_std)
print_scores(X_std, kmeans.labels_, "KMeans")
plot_clusters(X_std, kmeans.labels_, "KMeans")

# 2. Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3).fit(X_std)
print_scores(X_std, agg.labels_, "Agglomerative")
plot_clusters(X_std, agg.labels_, "Agglomerative")

# 3. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_std)
print_scores(X_std, dbscan.labels_, "DBSCAN")
plot_clusters(X_std, dbscan.labels_, "DBSCAN")

# 4. Spectral Clustering
spectral = SpectralClustering(n_clusters=3, random_state=42, assign_labels='discretize').fit(X_std)
print_scores(X_std, spectral.labels_, "Spectral")
plot_clusters(X_std, spectral.labels_, "Spectral")

# 5. Mean Shift
meanshift = MeanShift().fit(X_std)
print_scores(X_std, meanshift.labels_, "MeanShift")
plot_clusters(X_std, meanshift.labels_, "MeanShift")

plt.show()