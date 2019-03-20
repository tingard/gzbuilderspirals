from sklearn.cluster import DBSCAN


def cluster_arms(distances, eps=400, min_samples=4):
    return DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed',
        n_jobs=-1,
        algorithm='brute',
    ).fit(distances)
