import numpy as np
from sklearn.model_selection import GroupKFold
from . import metric, clustering, cleaning, deprojecting, fitting, pipeline
from . import xy_from_r_theta, r_theta_from_xy
from . import get_sample_weight, equalize_arm_length


class Arm():
    def __init__(self, parent_pipeline, arms, clean_points=True,
                 bar_length=0):
        self.arms = arms
        self.phi = parent_pipeline.phi
        self.ba = parent_pipeline.ba
        self.image_size = parent_pipeline.image_size

        bar_length = parent_pipeline.bar_length / self.image_size

        self.logsp_model = fitting.get_log_spiral_pipeline()

        self.coords, self.groups_all = cleaning.get_grouped_data(arms)
        self.deprojected_coords = deprojecting.deproject_arm(
            self.coords / self.image_size - 0.5,
            angle=self.phi, ba=self.ba,
        )
        self.R_all, self.t_all = r_theta_from_xy(*self.deprojected_coords.T)
        self.t_all_unwrapped = fitting.unwrap(self.t_all, self.R_all, self.groups_all)
        if clean_points:
            self.outlier_mask = cleaning.clean_arms_xy(
                self.coords,
                self.groups_all,
            )
        else:
            self.outlier_mask = np.ones(self.R_all.shape[0], dtype=bool)
        self.groups = self.groups_all[self.outlier_mask]
        self.R = self.R_all[self.outlier_mask]
        self.t = self.t_all_unwrapped[self.outlier_mask]
        self.point_weights = get_sample_weight(self.R, self.groups, bar_length)
        self.logsp_model.fit(self.t.reshape(-1, 1), self.R,
                             bayesianridge__sample_weight=self.point_weights)

        self.t_predict = np.linspace(min(self.t), max(self.t), 500)
        R_predict = self.logsp_model.predict(self.t_predict.reshape(-1, 1))

        t_predict = self.t_predict[R_predict > bar_length]
        R_predict = R_predict[R_predict > bar_length]

        x, y = xy_from_r_theta(R_predict, t_predict)
        self.length = np.sum(np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2))

        self.log_spiral = (np.stack((x, y), axis=1) + 0.5) * self.image_size

        self.reprojected_log_spiral = (deprojecting.reproject_arm(
            arm=np.stack((x, y), axis=1),
            angle=self.phi,
            ba=self.ba
        ) + 0.5) * self.image_size

        self.coef = self.logsp_model.named_steps['bayesianridge'].regressor_.coef_
        self.sigma = self.logsp_model.named_steps['bayesianridge'].regressor_.sigma_
        self.pa, self.sigma_pa = pipeline.get_pitch_angle(
            self.coef[0],
            self.sigma[0, 0]
        )

    def fit_polynomials(self, min_degree=1, max_degree=5, n_splits=5):
        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(self.groups))))
        models = {}
        scores = {}
        for degree in range(min_degree, max_degree):
            poly_model = fitting.get_polynomial_pipeline(degree)
            s = fitting.weighted_group_cross_val(
                poly_model,
                self.t.reshape(-1, 1), self.R,
                cv=gkf,
                groups=self.groups,
                weights=self.point_weights
            )
            poly_model.fit(self.t.reshape(-1, 1), self.R)
            poly_r = poly_model.predict(self.t_predict.reshape(-1, 1))
            models['poly_spiral_{}'.format(degree)] = np.vstack(self.t_predict, poly_r)
            scores['poly_spiral_{}'.format(degree)] = s
        return models, scores


class Pipeline():
    def __init__(self, drawn_arms, phi=0.0, ba=1.0, distances=None,
                 bar_length=10, image_size=512, parallel=False):
        self.drawn_arms = np.array(equalize_arm_length(np.array(drawn_arms)))
        self.image_size = image_size
        self.phi = phi
        self.ba = ba
        self.bar_length = bar_length
        if distances is None:
            cdm = (metric.calculate_distance_matrix_parallel if parallel
                   else metric.calculate_distance_matrix)
            if parallel:
                self.distances = cdm(self.drawn_arms)
            else:
                self.distances = cdm(self.drawn_arms)
        else:
            self.distances = distances
        self.db = clustering.cluster_arms(self.distances)

    def get_arm(self, arm_label, clean_points=True):
        arms_in_cluster = self.drawn_arms[self.db.labels_ == arm_label]
        return Arm(self, arms_in_cluster, clean_points=clean_points)

    def get_arms(self, merge=True, *args, **kwargs):
        if merge:
            return self.merge_arms([
                self.get_arm(i, *args, **kwargs)
                for i in range(max(self.db.labels_) + 1)
            ])
        return [
            self.get_arm(i, *args, **kwargs)
            for i in range(max(self.db.labels_) + 1)
        ]

    def merge_arms(self, arms, threshold=100, **kwargs):
        arms = np.array(arms)
        logsps = [arm.reprojected_log_spiral for arm in arms]
        pairs = []
        for i in range(len(logsps)):
            for j in range(i+1, len(logsps)):
                a, b = logsps[i], logsps[j]
                min_dist = min(
                    np.sum(metric._npsdtp_vfunc(a, b)) / len(a),
                    np.sum(metric._npsdtp_vfunc(b, a)) / len(b),
                )
                if min_dist <= threshold:
                    pairs.append([i, j])
        pairs = np.array(pairs)
        # we have a list of duplicate pairs, now check if we should merge more
        # than two arms at a time
        groups = []
        for i, pair in enumerate(pairs):
            if not np.any(np.isin(pair, groups)):
                groups.append(pair)
                continue
            for i in range(len(groups)):
                if np.any(np.isin(pair, groups[i])):
                    groups[i] = np.unique(np.concatenate((groups[i], pair)))
        groups += [[i] for i in range(len(arms)) if not i in pairs]
        merged_arms = []
        for group in groups:
            grouped_drawn_arms = np.concatenate([a.arms for a in arms[group]])
            new_arm = Arm(self, grouped_drawn_arms, **kwargs)
            merged_arms.append(new_arm)
        return np.array(merged_arms)
