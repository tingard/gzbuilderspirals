from sklearn.cluster import DBSCAN


def clusterArms(distanceMatrix):
    return DBSCAN(
        eps=20,
        min_samples=3,
        metric='precomputed',
        n_jobs=-1,
        algorithm='brute'
    ).fit(distanceMatrix)
