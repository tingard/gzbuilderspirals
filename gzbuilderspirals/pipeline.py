import numpy as np
from sklearn.model_selection import GroupKFold
from . import metric, clustering, cleaning, deprojecting, fitting
from . import r_theta_from_xy, get_sample_weight, equalize_arm_length, _vprint


def usage():
    print('This function accepts a numpy array of numpy arrays, each')
    print('of dimension (N, 2).')
    print('The available keyword arguments are ')
    print('imageSize: the size of the (square) galaxy image')
    print('phi: the galaxy\'s rotation in image coordinates')
    print('ba: the galaxy\'s axis ratio')
    print('verbose: whether to print logging information to the screen')


def _pipeline_iterator(
    drawn_arms, image_size=512, phi=0, ba=1, distances=None,
    clean_points=False,
):
    """Iterator to cycle through pipeline steps
    """
    drawn_arms = np.array(equalize_arm_length(np.array(drawn_arms)))
    yield drawn_arms  # Yield the equalised drawn arms
    if distances is None:
        distances = metric.calculate_distance_matrix(drawn_arms)
    # Yield the calculated distance matrix
    yield distances
    db = clustering.cluster_arms(distances)
    # Yield the labels of clustered arms
    yield db.labels_
    out = []
    for i in range(np.max(db.labels_) + 1):
        arms = drawn_arms[db.labels_ == i]
        coords, groups_all = cleaning.get_grouped_data(arms)
        deprojected_coords = deprojecting.deproject_arm(
            coords / image_size - 0.5,
            angle=phi, ba=ba,
        )
        # Yield the deprojected coordinates
        yield deprojected_coords
        R_all, t_all = r_theta_from_xy(*deprojected_coords.T)
        t_all_unwrapped = fitting.unwrap(t_all, groups_all)
        if clean_points:
            outlier_mask = cleaning.clean_arms_polar(
                R_all, t_all_unwrapped,
                groups_all
            )
        else:
            outlier_mask = np.ones(R_all.shape[0], dtype=bool)
        # Yield the outlier mask
        yield R_all, t_all_unwrapped, outlier_mask
        groups = groups_all[outlier_mask]
        R = R_all[outlier_mask].reshape(-1, 1)
        t = t_all_unwrapped[outlier_mask]
        point_weights = get_sample_weight(R.reshape(-1), groups)
        # Yield the point weights
        yield point_weights
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        logsp_model = fitting.get_polynomial_pipeline(1)
        out_ = {}
        s = fitting.weighted_group_cross_val(
            logsp_model,
            np.log(R), t,
            cv=gkf,
            groups=groups,
            weights=point_weights
        )
        out_['Log spiral'] = s
        for degree in range(1, 6):
            poly_model = fitting.get_polynomial_pipeline(degree)
            s = fitting.weighted_group_cross_val(
                poly_model,
                R, t,
                cv=gkf,
                groups=groups,
                weights=point_weights
            )
            out_['poly_spiral_{}'.format(degree)] = s
        # Yield the arm result
        yield out_
        out.append(out_)
    # yield out


def _pipeline_iterator2(
    drawn_arms, image_size=512, phi=0, ba=1, distances=None,
    clean_points=False,
):
    """Iterator to cycle through pipeline steps
    """
    drawn_arms = np.array(equalize_arm_length(np.array(drawn_arms)))
    yield drawn_arms  # Yield the equalised drawn arms
    if distances is None:
        distances = metric.calculate_distance_matrix(drawn_arms)
    # Yield the calculated distance matrix
    yield distances
    db = clustering.cluster_arms(distances)
    # Yield the labels of clustered arms
    yield db.labels_
    out = []
    for i in range(np.max(db.labels_) + 1):
        arms = drawn_arms[db.labels_ == i]
        coords, groups_all = cleaning.get_grouped_data(arms)
        deprojected_coords = deprojecting.deproject_arm(
            coords / image_size - 0.5,
            angle=phi, ba=ba,
        )
        # Yield the deprojected coordinates
        yield deprojected_coords
        R_all, t_all = r_theta_from_xy(*deprojected_coords.T)
        t_all_unwrapped = fitting.unwrap(t_all, groups_all)
        if clean_points:
            outlier_mask = cleaning.clean_arms_polar(
                R_all, t_all_unwrapped,
                groups_all
            )
        else:
            outlier_mask = np.ones(R_all.shape[0], dtype=bool)
        # Yield the outlier mask
        yield R_all, t_all_unwrapped, outlier_mask
        groups = groups_all[outlier_mask]
        R = R_all[outlier_mask]
        t = t_all_unwrapped[outlier_mask]
        point_weights = get_sample_weight(R, groups)
        # Yield the point weights
        yield point_weights
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        logsp_model = fitting.get_log_spiral_pipeline()
        out_ = {}
        s = fitting.weighted_group_cross_val(
            logsp_model,
            t.reshape(-1, 1), R,
            cv=gkf,
            groups=groups,
            weights=point_weights
        )
        out_['Log spiral'] = s
        for degree in range(1, 6):
            poly_model = fitting.get_polynomial_pipeline(degree)
            s = fitting.weighted_group_cross_val(
                poly_model,
                t.reshape(-1, 1), R,
                cv=gkf,
                groups=groups,
                weights=point_weights
            )
            out_['poly_spiral_{}'.format(degree)] = s
        # Yield the arm result
        yield out_
        out.append(out_)
    # yield out


def model_selection_pipeline(*args, verbose=False, **kwargs):
    gen = _pipeline_iterator(*args, **kwargs)
    v = verbose
    _vprint(v, '1 - equalising arm length')
    drawn_arms = next(gen)
    _vprint(v, '2 - calculating distance matrix')
    distances = next(gen)
    _vprint(v, '3 - clustering arms')
    labels = next(gen)
    out = []
    for i in range(np.max(labels) + 1):
        _vprint(v, 'Arm', i)
        _vprint(v, '\t1 - deprojecting arm')
        deprojected_coords = next(gen)
        _vprint(v, '\t2 - cleaning points')
        R_all, t_all_unwrapped, outlier_mask = next(gen)
        _vprint(v, '\t3 - calculating weights')
        point_weights = next(gen)
        _vprint(v, '\t4 - performing group cross-validation')
        out_ = next(gen)
        out.append(out_)
    return out


def msp(*args, **kwargs):
    for out in _pipeline_iterator(*args, **kwargs):
        pass
    return out
