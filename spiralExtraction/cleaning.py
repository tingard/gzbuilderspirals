from sklearn.neighbors import LocalOutlierFactor


def cleanPoints(pointCloud):
    clf = LocalOutlierFactor(n_neighbors=50)
    y_pred = clf.fit_predict(pointCloud)
    mask = ((y_pred + 1) / 2).astype(bool)
    return clf, mask
