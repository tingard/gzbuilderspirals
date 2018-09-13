import numpy as np
from scipy.interpolate import UnivariateSpline
import sys
from . import rThetaFromXY, fitSmoothedSpline
from . import deprojectArm
from . import metric
from . import clustering
from . import cleaning
from . import ordering


class GZBArm(object):
    def __init__(self, drawnArms):
        self.drawnArms = drawnArms
        self.pointCloud = np.array([
            point for arm in drawnArms
            for point in arm
        ])

    def clean(self):
        self.clf, self.outlierMask = cleaning.cleanPoints(self.pointCloud)
        self.cleanedCloud = self.pointCloud[self.outlierMask]
        return self.clf

    def chooseRepresentativeArm(self):
        try:
            self.armyArm = ordering.findArmyArm(self.drawnArms, self.clf)
        except AttributeError:
            self.clean()
            self.armyArm = ordering.findArmyArm(self.drawnArms, self.clf)
        return self.armyArm

    def orderAlongArm(self, arm=None, deviation=False):
        try:
            chosenArm = arm if arm else self.armyArm
        except AttributeError:
            self.chooseRepresentativeArm()
            chosenArm = arm if arm else self.armyArm

        self.deviationCloud = np.transpose(
            ordering.getDistAlongPolyline(self.cleanedCloud, chosenArm)
        )

        deviationEnvelope = np.abs(self.deviationCloud[:, 1]) < 30
        startEndMask = np.logical_and(
            self.deviationCloud[:, 0] > 0,
            self.deviationCloud[:, 0] < chosenArm.shape[0]
        )

        self.totalMask = np.logical_and(deviationEnvelope, startEndMask)

        self.pointOrder = np.argsort(self.deviationCloud[self.totalMask, 0])
        self.orderedPoints = self.cleanedCloud[self.totalMask][self.pointOrder]

        if deviation:
            return self.pointOrder, self.deviationCloud
        return self.pointOrder

    def fitSpline(self):
        t = self.deviationCloud[self.totalMask, 0][self.pointOrder].copy()
        t /= np.max(t)
        mask = t[1:] - t[:-1] <= 0
        while np.any(mask):
            mask = t[1:] - t[:-1] <= 0
            t[1:][mask] += 0.0001

        t = np.linspace(0, 1, self.orderedPoints.shape[0])
        Sx = UnivariateSpline(t, self.orderedPoints[:, 0], k=5)
        Sy = UnivariateSpline(t, self.orderedPoints[:, 1], k=5)
        return (Sx, Sy)

    def deproject(self, phi, ba, imageSize):
        deprojectedArms = np.array([
            deprojectArm(phi, ba, arm / imageSize - 0.5)
            for arm in self.drawnArms
        ])
        return GZBArm(deprojectedArms)

    def toRadial(self, points=None, mux=0, muy=0):
        if points is None:
            pointsToConvert = self.pointCloud[self.pointOrder]
        else:
            pointsToConvert = points
        return rThetaFromXY(
            *pointsToConvert.T,
            mux=0, muy=0
        )


class GalaxySpirals(object):
    def __init__(self, drawnArms, phi=0, ba=1, imageSize=512):
        self.drawnArms = drawnArms
        self.ba = ba
        self.phi = phi
        self.imageSize = imageSize
        self.deprojectedArms = np.array([
            deprojectArm(phi, ba, arm / imageSize - 0.5)
            for arm in drawnArms
        ])

    def calculateDistances(self):
        self.distances = metric.calculateDistanceMatrix(self.drawnArms)
        return self.distances

    def clusterLines(self, distances=None, redoDistances=False):
        if distances is not None:
            self.distances = distances
            self.db = clustering.clusterArms(distances)
        else:
            try:
                self.db = clustering.clusterArms(self.distances)
            except AttributeError:
                self.distances = metric.calculateDistanceMatrix(self.drawnArms)
                self.db = clustering.clusterArms(self.distances)
            self.arms = [
                GZBArm(self.drawnArms[self.db.labels_ == i])
                for i in range(np.max(self.db.labels_) + 1)
            ]
        return self.db

    def fitArms(self):
        for arm in self.arms:
            arm.clean()
            arm.chooseRepresentativeArm()
            arm.orderAlongArm()


def test():
    a = np.stack(
        (
            np.tile(np.arange(100), 5).reshape(5, 100),
            np.array([
                np.random.random(size=100) for i in range(5)
            ]).reshape(5, 100)
        ),
        axis=2
    )
    b = np.stack(
        (
            np.tile(np.arange(100), 5).reshape(5, 100),
            np.array([
                np.random.random(size=100) + 100 for i in range(5)
            ]).reshape(5, 100)
        ),
        axis=2
    )
    arms = np.concatenate((a, b))
    S = GalaxySpirals(arms, imageSize=100)
    print('Clustering:')
    db = S.clusterLines()
    print('Passed test?', np.all(db.labels_ == ([0] * 5 + [1] * 5)))
