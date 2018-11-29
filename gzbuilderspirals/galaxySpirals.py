import numpy as np
from scipy.interpolate import splprep, splev, interp1d
from . import r_theta_from_xy
from . import metric
from . import clustering
from . import cleaning
from . import ordering
from . import fitting
from . import deprojecting


class GZBArm(object):
    def __init__(self, drawn_arms, image_size=512, equalize_arms=True):
        self.drawn_arms = (
            self.equalize_arm_length(drawn_arms, method=np.average)
            if equalize_arms
            else drawn_arms
        )
        self.point_cloud = np.array([
            point
            for arm in self.drawn_arms
            for point in arm
        ])
        self.image_size = image_size
        self.r_weight_func = self.weight_r_by_n_arms(self.drawn_arms)
        self.clean()

    def equalize_arm_length(self, arms, method=np.max):
        new_you = np.linspace(0, 1, method([len(i) for i in arms]))
        return [
            np.array(splev(new_you, splprep(arm.T, s=0, k=1)[0])).T
            for arm in arms
        ]

    def normalise(self, a):
        return (a / self.image_size) - 0.5

    def de_normalise(self, a):
        return (a + 0.5) * self.image_size

    def clean(self):
        self.clf, self.outlier_mask = cleaning.clean_points(self.point_cloud)
        self.cleaned_cloud = self.point_cloud[self.outlier_mask]
        return self.clf

    def order_along_polyline(self, arm):
        result = {}
        result['deviation_cloud'] = np.transpose(
            ordering.get_dist_along_polyline(self.cleaned_cloud, arm)
        )

        deviation_envelope = np.abs(result['deviation_cloud'][:, 1]) < 30
        start_end_mask = np.logical_and(
            result['deviation_cloud'][:, 0] > 0,
            result['deviation_cloud'][:, 0] < arm.shape[0]
        )

        total_mask = np.logical_and(deviation_envelope, start_end_mask)

        unmasked_order = np.argsort(
            result['deviation_cloud'][total_mask, 0]
        )
        result['point_order'] = np.where(total_mask)[0][unmasked_order]
        result['ordered_points'] = (
            self.cleaned_cloud[result['point_order']]
        )
        return result

    def deproject(self, phi, ba):
        def dp(arr):
            return self.de_normalise(deprojecting.deproject_arm(
                phi, ba,
                self.normalise(arr)
            ))

        deprojected = GZBArm(np.array([
            dp(arm)
            for arm in self.drawn_arms
        ]), equalize_arms=False)
        try:
            deprojected.point_cloud = dp(self.point_cloud)
            deprojected.outlier_mask = self.outlier_mask
            deprojected.cleaned_cloud = (
                deprojected.point_cloud[self.outlier_mask]
            )

        except AttributeError:
            pass
        return deprojected

    def unwrap_and_sort(self):
        # grab r and theta for each arm in the cluster
        drawn_arms_r_theta = [
            r_theta_from_xy(*self.normalise(a).T)
            for a in self.drawn_arms
        ]
        # grab R array
        R = np.fromiter(
            (j for i in drawn_arms_r_theta for j in np.sort(i[0])),
            dtype=float
        )
        # construct theta array
        t = np.array([])
        # for each arm cluster...
        for i, (r, theta) in enumerate(drawn_arms_r_theta):
            # unwrap the drawn arm
            r_, t_ = fitting.unwrap_and_sort(r, theta)
            # set the minimum theta of each arm to be in [0, 2pi) (not true in
            # general but should pop out as an artefact of the clustering alg)
            while np.min(t_) < 0:
                t_ += 2*np.pi
            # add this arm to the theta array
            t = np.concatenate((t, t_))
        # sort the resulting points by radius
        a = np.argsort(R)
        return np.stack((R[a], t[a]))

    def weight_r_by_n_arms(self, drawn_arms):
        radiuses = [
            r_theta_from_xy(
                *self.normalise(a).T
            )[0]
            for a in drawn_arms
        ]

        r_bins = np.linspace(np.min(radiuses), np.max(radiuses), 100)
        counts = np.zeros(r_bins.shape)
        for i, _ in enumerate(r_bins[:-1]):
            n = sum(
                1
                for r in radiuses
                if np.any(r > r_bins[i]) and np.any(r < r_bins[i+1])
            )
            counts[i] = max(0, n)
        return interp1d(r_bins, counts/sum(counts))

    def get_sample_weight(self, R):
        w = R**2
        w *= self.weight_r_by_n_arms(self.drawn_arms)(R)
        w /= np.average(w)
        return w

    def fit(self, **kwargs):
        R, t = self.unwrap_and_sort()
        sample_weight = self.get_sample_weight(R)
        self.fit_result = fitting.fit_to_points(
            *self.unwrap_and_sort(),
            sample_weight=sample_weight,
            **kwargs
        )
        return self.fit_result

    def reproject(self, phi, ba, arm):
        return deprojecting.reproject_arm(phi, ba, arm)

    def reproject_fit(self, phi, ba):
        try:
            return {
                'log_spiral': self.de_normalise(
                    deprojecting.reproject_arm(
                        phi, ba,
                        self.fit_result['xy_fit']['log_spiral']
                    )
                ),
                'polynomial': self.de_normalise(
                    deprojecting.reproject_arm(
                        phi, ba,
                        self.fit_result['xy_fit']['polynomial']
                    )
                )
            }
        except AttributeError:
            self.fit()
            return {
                'log_spiral': self.de_normalise(
                    deprojecting.reproject_arm(
                        phi, ba,
                        self.fit_result['xy_fit']['log_spiral']
                    )
                ),
                'polynomial': self.de_normalise(
                    deprojecting.reproject_arm(
                        phi, ba,
                        self.fit_result['xy_fit']['polynomial']
                    )
                )
            }


class GalaxySpirals(object):
    def __init__(self, drawn_arms, phi=0, ba=1, image_size=512):
        self.drawn_arms = drawn_arms
        self.ba = ba
        self.phi = phi
        self.image_size = image_size

    def calculate_distances(self):
        self.distances = metric.calculate_distance_matrix(self.drawn_arms)
        return self.distances

    def cluster_lines(self, distances=None, redo_distances=False):
        if distances is not None:
            self.distances = distances
        try:
            self.db = clustering.cluster_arms(self.distances)
        except AttributeError:
            self.distances = metric.calculate_distance_matrix(self.drawn_arms)
            self.db = clustering.cluster_arms(self.distances)
        self.arms = [
            GZBArm(self.drawn_arms[self.db.labels_ == i], self.image_size)
            for i in range(np.max(self.db.labels_) + 1)
        ]
        return self.db

    def deproject_arms(self):
        self.deprojected_arms = [
            arm.deproject(self.phi, self.ba)
            for arm in self.arms
        ]
        return self.deprojected_arms

    def fit_arms(self, **kwargs):
        try:
            return [
                arm.reproject_fit(self.phi, self.ba)
                for arm in self.deprojected_arms
            ]
        except AttributeError:
            self.deproject_arms()
            return [
                arm.reproject_fit(self.phi, self.ba)
                for arm in self.deprojected_arms
            ]
