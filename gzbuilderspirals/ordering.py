import numpy as np
from scipy.interpolate import UnivariateSpline, splprep, splev
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from .metric import vCalcT


def findArmyArm(arms, clf, smooth=True):
    i = np.argmax([
        np.sum(clf._decision_function(arm)) / arm.shape[0]
        for arm in arms
    ])
    arm = arms[i]
    if not smooth:
        return arm

    t = np.linspace(0, 1, arm.shape[0])

    Sx = UnivariateSpline(t, arm[:, 0], s=512, k=5)
    Sy = UnivariateSpline(t, arm[:, 1], s=512, k=5)

    smoothedArm = np.stack((Sx(t), Sy(t)), axis=1)

    return smoothedArm


def sign(a):
    b1 = a[:, 0, :] - a[:, 1, :]
    b2 = a[:, 2, :] - a[:, 1, :]
    paddedB1 = np.pad(b1, ((0, 0), (0, 1)), 'constant', constant_values=(0,))
    paddedB2 = np.pad(b2, ((0, 0), (0, 1)), 'constant', constant_values=(0,))
    s = np.sign(np.cross(paddedB1, paddedB2, axisa=1, axisb=1))[:, 2]
    s[s == 0] = 1.0
    return s


def getDiff2(t, a):
    projection = a[:, 1, :] + np.repeat(
        t.reshape(-1, 1), 2, axis=1) * (a[:, 2, :] - a[:, 1, :])
    outsideBounds = np.logical_or(t < 0, t > 1)
    out = np.add.reduce(
        (a[:, 0, :] - projection) * (a[:, 0, :] - projection),
        axis=1
    )
    endPointDistance = np.amin([
        np.add.reduce((a[outsideBounds, 1] - a[outsideBounds, 0])**2, axis=1),
        np.add.reduce((a[outsideBounds, 2] - a[outsideBounds, 0])**2, axis=1)
    ], axis=0)
    # If we have gone beyond endpoints, set distance to be the distance to the
    # end point (rather than to a continuation of the line)
    out[outsideBounds] = endPointDistance
    return np.sqrt(out)


vGetDiff2 = np.vectorize(getDiff2, signature='(a),(a,b,c)->(a)')
vSign = np.vectorize(sign, signature='(a,b,c)->(a)')


def getDistAlongPolyline(points, polyLine):
    # construct our tensor (allowing vectorization)
    # m{i, j, k, p}
    # i iterates over each point in a
    # j cycles through each pair of points in b
    # k cycles through (a[i], b[j], b[j+1])
    # p represents [x, y]
    m = np.zeros((points.shape[0], polyLine.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(points, [m.shape[1] + 1, 1, 1]), axes=[1, 0, 2]
    )[:, :-1, :]
    m[:, :, 1, :] = np.tile(polyLine, [points.shape[0], 1, 1])[:, :-1, :]
    m[:, :, 2, :] = np.tile(
        np.roll(polyLine, -1, axis=0), [points.shape[0], 1, 1]
    )[:, :-1, :]

    t = vCalcT(m)
    signs = vSign(m)
    distances = vGetDiff2(t, m)
    minDistIndex = np.argmin(distances, axis=1)
    optimumIndex = np.dstack(
        (np.arange(minDistIndex.shape[0]), minDistIndex)
    )[0]
    return (
        minDistIndex + t[optimumIndex[:, 0], optimumIndex[:, 1]],
        (
            distances[optimumIndex[:, 0], optimumIndex[:, 1]]
            * signs[optimumIndex[:, 0], optimumIndex[:, 1]]
        )
    )


def getOrderedClusters(X):
    knn_graph = kneighbors_graph(X, 30, include_self=False)

    model = AgglomerativeClustering(
        linkage='ward',
        connectivity=knn_graph,
        n_clusters=10
    )
    model.fit(X)
    means = np.array([
        np.mean(X[model.labels_ == l], axis=0)
        for l in range(max(model.labels_) + 1)
    ])
    nn = NearestNeighbors(3, algorithm='kd_tree')
    nn.fit(means)

    foo = np.array([nn.kneighbors([p], 3, True) for p in means])[:, :, :, 1:]

    # find an endpoint
    endpoint = np.floor_divide(np.argsort(foo[:, 0].reshape(-1)), 2)[-1]
    endpoint, foo
    bar = [endpoint]
    # iterate till end
    for i in range(len(means) - 1):
        dist, j = foo[int(bar[-1])].reshape(2, -1)
        j = j.astype(int)
        if j[0] in bar and j[1] in bar:
            break
        if j[0] in bar:
            bar.append(j[1])
        elif j[1] in bar:
            bar.append(j[0])
        else:
            bar.append(j[np.argsort(dist)[0]])
    return means[bar]


def getSortingLine(normalisedMeans):
    # mask to ensure no points are in the same place
    separationMask = np.ones(normalisedMeans.shape[0], dtype=bool)
    separationMask[1:] = np.linalg.norm(
        normalisedMeans[1:] - normalisedMeans[:-1],
        axis=1
    ) != 0
    # perform the interpolation
    tck, u = splprep(normalisedMeans[separationMask].T, s=0.01, k=3)

    unew = np.linspace(0, 1, 500)

    return np.array(splev(unew, tck)).T
