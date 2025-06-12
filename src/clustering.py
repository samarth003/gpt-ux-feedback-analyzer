from sklearn.cluster import KMeans

def cluster_feedback(X, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans