import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
import sys
from . import deprojectArm
import metric
import clustering
import cleaning
import ordering


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
            for arm in drawnArms
        ])
        return GZBArm(deprojectedArms)

    def toRadial(self, points=None, mux=0, muy=0):
        if points is None:
            pointsToConvert = self.pointCloud
        else:
            pointsToConvert = points
        return spiralExtraction.rThetaFromXY(
            normalisedPoints[:, 0],
            normalisedPoints[:, 1],
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
        if not self.distances or redoDistances:
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


if __name__ == "__main__":
    a = np.stack(
        (
            np.tile(np.arange(100), 5),
            np.tile(np.random.random(size=100), 5)
        ),
        axis=1
    )
    print(a)
