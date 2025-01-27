from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class Clustering:
    @staticmethod
    def kmeans_clustering(embeddings, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels

    @staticmethod
    def gmm_clustering(embeddings, n_clusters):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(embeddings)
        return labels

    @staticmethod
    def evaluate_clustering(cluster_labels, ground_truth):
        ari = adjusted_rand_score(ground_truth, cluster_labels)
        nmi = normalized_mutual_info_score(ground_truth, cluster_labels)
        return {"ARI": ari, "NMI": nmi}
