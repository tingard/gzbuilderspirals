import numpy as np
from scipy.interpolate import splprep, splev, UnivariateSpline, interp1d
from . import rThetaFromXY, xyFromRTheta, deprojectArm
from . import metric
from . import clustering
from . import cleaning
from . import ordering
from . import fitting

class GZBArm(object):
    def __init__(self, drawnArms, imageSize=512):
        self.drawnArms = drawnArms
        self.pointCloud = np.array([
            point for arm in drawnArms
            for point in arm
        ])
        self.imageSize = imageSize
        self.clean()

    def normalise(self, a):
        return (a / self.imageSize) - 0.5

    def deNorm(self, a):
        return (a + 0.5) * self.imageSize

    def clean(self):
        self.clf, self.outlierMask = cleaning.cleanPoints(self.pointCloud)
        self.cleanedCloud = self.pointCloud[self.outlierMask]
        return self.clf

    def getSortingLine(self, splineKwargs={}):
        # cluster the arm into groups, then order these groups using their
        # nearest neighbors
        means = ordering.getOrderedClusters(self.cleanedCloud)
        # get a smoothed line from these points to order the cloud along
        return self.deNorm(
            ordering.getSortingLine(self.normalise(means))
        )

    def orderAlongPolyLine(self, arm):
        result = {}
        result['deviationCloud'] = np.transpose(
            ordering.getDistAlongPolyline(self.cleanedCloud, arm)
        )

        deviationEnvelope = np.abs(result['deviationCloud'][:, 1]) < 30
        startEndMask = np.logical_and(
            result['deviationCloud'][:, 0] > 0,
            result['deviationCloud'][:, 0] < arm.shape[0]
        )

        totalMask = np.logical_and(deviationEnvelope, startEndMask)

        unMaskedOrder = np.argsort(
            result['deviationCloud'][totalMask, 0]
        )
        result['pointOrder'] = np.where(totalMask)[0][unMaskedOrder]
        result['orderedPoints'] = (
            self.cleanedCloud[result['pointOrder']]
        )
        return result

    def deproject(self, phi, ba):
        dp = lambda arr: self.deNorm(deprojectArm(
            phi, ba,
            self.normalise(arr)
        ))

        deprojected = GZBArm(np.array([
            dp(arm)
            for arm in self.drawnArms
        ]))
        try:
            deprojected.pointCloud = dp(self.pointCloud)
            deprojected.outlierMask = self.outlierMask
            deprojected.cleanedCloud = (
                deprojected.pointCloud[self.outlierMask]
            )

        except AttributeError:
            pass
        return deprojected

    def fit(self, **kwargs):
        return fitting.fitToPoints(self, **kwargs)


class GalaxySpirals(object):
    def __init__(self, drawnArms, phi=0, ba=1, imageSize=512):
        self.drawnArms = drawnArms
        self.ba = ba
        self.phi = phi
        self.imageSize = imageSize

    def calculateDistances(self):
        self.distances = metric.calculateDistanceMatrix(self.drawnArms)
        return self.distances

    def clusterLines(self, distances=None, redoDistances=False):
        if distances is not None:
            self.distances = distances
        try:
            self.db = clustering.clusterArms(self.distances)
        except AttributeError:
            self.distances = metric.calculateDistanceMatrix(self.drawnArms)
            self.db = clustering.clusterArms(self.distances)
        self.arms = [
            GZBArm(self.drawnArms[self.db.labels_ == i], self.imageSize)
            for i in range(np.max(self.db.labels_) + 1)
        ]
        return self.db

    def deprojectArms(self):
        return [
            arm.deproject(self.phi, self.ba)
            for arm in self.arms
        ]

    def fitArms(self, **kwargs):
        return [
            arm.fit(**kwargs)
            for arm in self.deprojectArms()
        ]
