import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
import sys
from . import rThetaFromXY
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

        deviationCloud = np.transpose(
            ordering.getDistAlongPolyline(self.cleanedCloud, chosenArm)
        )

        deviationEnvelope = np.abs(deviationCloud[:, 1]) < 30
        startEndMask = np.logical_and(
            deviationCloud[:, 0] > 0,
            deviationCloud[:, 0] < chosenArm.shape[0]
        )

        totalMask = np.logical_and(deviationEnvelope, startEndMask)

        self.pointOrder = np.argsort(deviationCloud[totalMask, 0])

        if deviation:
            return self.pointOrder, deviationCloud
        return self.pointOrder

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

    def clusterLines(self, redoDistances=False):
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
