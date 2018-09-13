from sklearn.cluster import DBSCAN


def clusterArms(distances):
    return DBSCAN(
        eps=400,
        min_samples=5,
        metric='precomputed',
        n_jobs=-1,
        algorithm='brute'
    ).fit(distances)
