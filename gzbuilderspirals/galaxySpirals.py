import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
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
        t /= np.max(t)
        # ensure it's strictly monotonically increasing
        mask = t[1:] - t[:-1] <= 0
        while np.any(mask):
            mask = t[1:] - t[:-1] <= 0
            t[1:][mask] += 0.0001
        t /= np.max(t)
        return t

    def fitXYSpline(self, ordering, t):
        normalisedPoints = (ordering.orderedPoints / self.imageSize - 0.5)
        Sx = UnivariateSpline(t, normalisedPoints[:, 0], k=5, s=0.25)
        Sy = UnivariateSpline(t, normalisedPoints[:, 1], k=5, s=0.25)
        return (Sx, Sy)

    def deproject(self, phi, ba):
        dp = lambda arr: (deprojectArm(
            phi, ba,
            arr / self.imageSize - 0.5
        ) + 0.5) * self.imageSize

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
            self.db = clustering.clusterArms(distances)
        else:
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

    def fitArms(self, t=np.linspace(0, 1, 1000)):
        result = SimpleNamespace(
            xySplines=[],
            deprojectedArms=[],
            radialFit=[],
        )
        for arm in self.arms:
            arm.clean()
            representativeArm = arm.chooseRepresentativeArm()
            ordering = arm.orderAlongArm(representativeArm)

            chosenT = arm.genTFromOrdering(ordering)

            Sx, Sy = arm.fitXYSpline(ordering, chosenT)
            imageSpline = np.stack((Sx(t), Sy(t)), axis=1)

            deprojectedSpline = deprojectArm(
                self.phi, self.ba,
                imageSpline
            )

            deprojectedArm = arm.deproject(self.phi, self.ba)
            ordering2 = deprojectedArm.orderAlongArm(
                arm.deNorm(deprojectedSpline)
            )

            deprojectedT = deprojectedArm.genTFromOrdering(ordering2)
            orderedDeprojectedCloud = (
                deprojectedArm.cleanedCloud[ordering2.pointOrder]
            )
            r, theta = rThetaFromXY(
                *orderedDeprojectedCloud.T / 512 - 0.5,
                mux=0, muy=0
            )
            aaR, aaTh = rThetaFromXY(
                *(deprojectedSpline.T)
            )

            tFunc = interp1d(t, aaTh)

            Sr = UnivariateSpline(deprojectedT, r, k=5)
            xr, yr = xyFromRTheta(Sr(t), tFunc(t), mux=0, muy=0)

            result.xySplines.append([Sx, Sy])
            result.deprojectedArms.append(deprojectedArm)
            result.radialFit.append(np.stack((xr, yr), axis=1))
        return result


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
