import numpy as np
from scipy.interpolate import splprep, splev, UnivariateSpline, interp1d
from types import SimpleNamespace
from . import rThetaFromXY, xyFromRTheta, deprojectArm
from . import metric
from . import clustering
from . import cleaning
from . import ordering


class GZBArm(object):
    def __init__(self, drawnArms, imageSize=512):
        self.drawnArms = drawnArms
        self.pointCloud = np.array([
            point for arm in drawnArms
            for point in arm
        ])
        self.imageSize = imageSize

    def normalise(self, a):
        return (a / self.imageSize) - 0.5

    def deNorm(self, a):
        return (a + 0.5) * self.imageSize

    def clean(self):
        self.clf, self.outlierMask = cleaning.cleanPoints(self.pointCloud)
        self.cleanedCloud = self.pointCloud[self.outlierMask]
        return self.clf

    def chooseRepresentativeArm(self):
        try:
            armyArm = ordering.findArmyArm(self.drawnArms, self.clf)
        except AttributeError:
            self.clean()
            armyArm = ordering.findArmyArm(self.drawnArms, self.clf)
        return armyArm

    def orderAlongArm(self, arm):
        result = SimpleNamespace()
        result.deviationCloud = np.transpose(
            ordering.getDistAlongPolyline(self.cleanedCloud, arm)
        )

        deviationEnvelope = np.abs(result.deviationCloud[:, 1]) < 30
        startEndMask = np.logical_and(
            result.deviationCloud[:, 0] > 0,
            result.deviationCloud[:, 0] < arm.shape[0]
        )

        totalMask = np.logical_and(deviationEnvelope, startEndMask)

        unMaskedOrder = np.argsort(
            result.deviationCloud[totalMask, 0]
        )
        result.pointOrder = np.where(totalMask)[0][unMaskedOrder]
        result.orderedPoints = (
            self.cleanedCloud[result.pointOrder]
        )
        return result

    def genTFromOrdering(self, ordering):
        t = ordering.deviationCloud[ordering.pointOrder, 0].copy()
        # ensure it's strictly monotonically increasing
        mask = t[1:] - t[:-1] <= 0
        while np.any(mask):
            mask = t[1:] - t[:-1] <= 0
            t[1:][mask] += 0.0001

        # normalise from 0 to 1
        t -= np.min(t)
        t /= np.max(t)
        return t

    def fitSpline(self, ordering):
        normalisedPoints = self.normalise(ordering.orderedPoints)
        # mask to ensure no points are in the same place
        pm = np.ones(normalisedPoints.shape[0], dtype=bool)
        pm[1:] = np.linalg.norm(
            normalisedPoints[1:] - normalisedPoints[:-1],
            axis=1
        ) != 0
        # perform the interpolation
        tck, u = splprep(normalisedPoints[pm].T, s=1, k=4)
        return np.array(splev(u, tck)).T

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
            deprojected.cleanedCloud = (
                deprojected.pointCloud[self.outlierMask]
            )

        except AttributeError:
            pass
        return deprojected


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

    def fitXYSplines(self):
        result = SimpleNamespace(
            representativeArms=[],
            orderings=[],
            fits=[],
        )
        for arm in self.arms:
            arm.clean()
            representativeArm = arm.chooseRepresentativeArm()
            ordering = arm.orderAlongArm(representativeArm)

            fit = arm.fitSpline(ordering)

            result.representativeArms.append(representativeArm)
            result.orderings.append(ordering)
            result.fits.append(fit)
        return result

    def fitRadialSplines(self, xyResult):
        result = SimpleNamespace(
            deprojectedArms=[],
            radialFit=[],
            orderings=[],
        )
        for i, arm in enumerate(self.arms):
            xyFit = xyResult.fits[i]
            t = np.linspace(0, 1, xyFit.shape[0])

            deprojectedXYFit = deprojectArm(
                self.phi, self.ba,
                xyFit
            )

            deprojectedArm = arm.deproject(self.phi, self.ba)
            ordering = deprojectedArm.orderAlongArm(
                arm.deNorm(deprojectedXYFit)
            )

            deprojectedT = deprojectedArm.genTFromOrdering(ordering)
            orderedDeprojectedCloud = (
                deprojectedArm.cleanedCloud[ordering.pointOrder]
            )
            r, theta = rThetaFromXY(
                *arm.normalise(orderedDeprojectedCloud).T,
                mux=0, muy=0
            )
            aaR, aaTh = rThetaFromXY(
                *(deprojectedXYFit.T)
            )

            tFunc = interp1d(t, aaTh)

            Sr = UnivariateSpline(deprojectedT, r, k=4, s=1)
            xr, yr = xyFromRTheta(Sr(t), tFunc(t), mux=0, muy=0)

            result.deprojectedArms.append(deprojectedArm)
            result.orderings.append(ordering)
            result.radialFit.append(np.stack((xr, yr), axis=1))
        return result
