import numpy as np
from scipy.interpolate import UnivariateSpline
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
