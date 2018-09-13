import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
import sys
from . import log
import metric
import clustering
import cleaning
import ordering


def fitSmoothedSpline(points):
    t = np.linspace(0, 1, points.shape[0])
    Sx = UnivariateSpline(t, points[:, 0], k=5)#, s=0.25)
    Sy = UnivariateSpline(t, points[:, 1], k=5)#, s=0.25)
    return (Sx, Sy)


def usage():
    print('This function accepts a numpy array of numpy arrays, each')
    print('of dimension (N, 2).')
    print('The available keyword arguments are ')
    print('imageSize: the size of the (square) galaxy image')
    print('phi: the galaxy\'s rotation in image coordinates')
    print('ba: the galaxy\'s axis ratio')
    print('verbose: whether to print logging information to the screen')


def fit(
    drawnArms, imageSize=512, verbose=True, fullOutput=True, phi=0, ba=1
):
    if not isinstance(drawnArms, np.array):
        usage()
        sys.exit(0)
    log('Calculating distance matrix (this can be slow)', verbose)
    functions = []
    clfs = []
    distances = metric.calculateDistanceMatrix(drawnArms)
    log('Clustering arms', verbose)
    db = clustering.clusterArms(distances)

    for label in np.unique(db.labels_):
        if label < 0:
            continue
        log('Working on arm label {}'.format(label), verbose)
        pointCloud = np.array([
            point for arm in drawnArms[db.labels_ == label]
            for point in arm
        ])
        log(
            '\t[1 / 4] Cleaning points ({} total)'.format(pointCloud.shape[0]),
            verbose
        )
        clf, mask = cleaning.cleanPoints(pointCloud)
        clfs.append(clf)
        cleanedCloud = pointCloud[mask]
        log('\t[2 / 4] Identifiying most representitive arm', verbose)
        armyArm = ordering.findArmyArm(drawnArms[db.labels_ == label], clf)
        log('\t[3 / 4] Sorting points', verbose)
        deviationCloud = np.transpose(
            ordering.getDistAlongPolyline(cleanedCloud, armyArm)
        )

        deviationEnvelope = np.abs(deviationCloud[:, 1]) < 30
        startEndMask = np.logical_and(
            deviationCloud[:, 0] > 0,
            deviationCloud[:, 0] < armyArm.shape[0]
        )

        totalMask = np.logical_and(deviationEnvelope, startEndMask)

        pointOrder = np.argsort(deviationCloud[totalMask, 0])

        normalisedPoints = cleanedCloud[totalMask][pointOrder] / imageSize
        normalisedPoints -= 0.5

        log('\t[4 / 4] Fitting Spline', verbose)
        t = np.linspace(0, 1, normalisedPoints.shape[0])
        t = deviationCloud[totalMask, 0][pointOrder].copy()
        t /= np.max(t)
        mask = t[1:] - t[:-1] <= 0
        while np.any(mask):
            mask = t[1:] - t[:-1] <= 0
            t[1:][mask] += 0.0001

        Sx = UnivariateSpline(t, normalisedPoints[:, 0], k=5)
        Sy = UnivariateSpline(t, normalisedPoints[:, 1], k=5)

        functions.append([Sx, Sy])
    log('done!', verbose)
    if not fullOutput:
        return functions

    returns = {
        'functions': functions,
        'distances': distances,
        'LOF': clfs,
        'labels': db.labels_
    }
    return returns
