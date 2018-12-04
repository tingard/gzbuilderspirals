import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import median_absolute_error
from gzbuilderspirals import r_theta_from_xy, xy_from_r_theta

# obtained by fitting a semi-truncated Gamma distribution to spiral arm width
# slider values, after removing all values at 1 (default) and 0 and 2
# (extremes)
alpha_1_prior = 1.0230488753392505e-08
alpha_2_prior = 0.6923902410146074
clf_kwargs = {}
clf_kwargs.setdefault('alpha_1', alpha_1_prior)
clf_kwargs.setdefault('alpha_2', alpha_2_prior)


def AIC(log_likelihood, k=1):
    """Compute the AIC for a fit"""
    return 2 * k - 2 * log_likelihood


def AICc(score, k=1, n=-1):
    """Compute the AICc for a fit"""
    return (
        AIC(score, k=k)
        + (2 * k**2 + 2 * k) / (n - k - 1)
    )


def BIC(score, k=1, n=-1):
    """Compute the BIC for a fit"""
    bic = k * np.log(n) - 2 * score
    return bic


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def unwrap_and_sort(r, theta):
    """Given r and theta, try and order points along the spiral
    -> Sort the points radially
    -> Use this sorting to unwrap theta (yay numpy)
    """
    a = np.argsort(r)
    theta_ordered = np.unwrap(theta[a])
    r_ordered = r[a]
    return r_ordered, theta_ordered


def unwrap(theta, groups):
    t = np.array([])
    dt = (np.arange(5) - 2) * 2 * np.pi
    theta_mean = 0
    for i, g in enumerate(np.unique(groups)):
        t_ = np.unwrap(theta[groups == g])
        # out of all allowed transformations, which puts the mean of the theta
        # values closest to the mean of rest of the points (using the first arm
        # as a template)?
        if i == 0:
            theta_mean = np.concatenate((t, t_)).mean()
        j = np.argmin(np.abs(t_.mean() + dt - theta_mean))
        t_ += dt[j]
        t = np.concatenate((t, t_))
    return t


def log_spiral_fit(R, t, prediction_resolution=100, sample_weight=None,
                   clf_kwargs={}):
    """Given the (unwrapped and ordered) theta and r values of
    cleaned points in a spiral, fit a logarithmic spiral using
    BayesianRidge regression. We actually fit
    theta = ⍺ * log(r) + β
    where ⍺ = 1 / b, β = log(a) / b
    (r = a * exp(b * theta) for a standard log spiral profile)
    """
    Rp = np.log(R)
    Rp_average = np.average(Rp, axis=0)
    t_average = np.average(t, axis=0)
    t_centered = t - t_average
    Rp_centered = Rp - Rp_average
    point_weights = (
        R**2 / np.average(R**2)
        if sample_weight is None
        else sample_weight
    )

    clf_kwargs.setdefault('alpha_1', alpha_1_prior)
    clf_kwargs.setdefault('alpha_2', alpha_2_prior)

    clf = BayesianRidge(
        compute_score=True,
        fit_intercept=True,
        copy_X=True,
        normalize=True,
        **clf_kwargs
    )

    Rp_predict = np.linspace(min(Rp_centered), max(Rp_centered),
                             prediction_resolution)

    X = np.vander(Rp_centered, 2)
    X_predict = np.vander(Rp_predict, 2)
    clf.fit(X[:, :-1], t_centered, sample_weight=point_weights)
    T, T_std = clf.predict(
        X_predict[:, :-1],
        return_std=True
    )
    return {
        'AIC': AICc(clf.scores_[-1], 3, T.shape[0]),
        'BIC': BIC(clf.scores_[-1], 3, T.shape[0]),
        'clf': clf,
        'k': 3,
        'R': np.exp(Rp_predict + Rp_average),
        'T': T + t_average,
        'T_std': T_std,
    }


def polynomial_fit(R, t, degree=None, prediction_resolution=100,
                   sample_weight=None, clf_kwargs={}):
    r"""Given the (unwrapped and ordered) theta and r values of
    cleaned points in a spiral, fit a polynomial of r to theta.
    If degree not specified use ARDRegression, otherwise use
    BayesianRidge.
    Also accepts an R_weighting array
    """
    R_average = np.average(R, axis=0)
    t_average = np.average(t, axis=0)
    t_centered = t - t_average
    R_centered = R - R_average
    point_weights = (
        R**2 / np.average(R**2)
        if sample_weight is None
        else sample_weight
    )

    clf_kwargs.setdefault('alpha_1', alpha_1_prior)
    clf_kwargs.setdefault('alpha_2', alpha_2_prior)

    R_predict = np.linspace(min(R_centered), max(R_centered),
                            prediction_resolution)
    degree_ = degree if degree is not None else 50
    X = np.vander(R_centered, degree_)
    X_predict = np.vander(R_predict, degree_)

    if degree is None:
        clf = ARDRegression(
            compute_score=True,
            fit_intercept=True,
            copy_X=True,
            normalize=True,
            **clf_kwargs
        )
        clf.fit(X[:, :-1], t_centered)
    else:
        clf = BayesianRidge(
            compute_score=True,
            fit_intercept=True,
            copy_X=True,
            normalize=True,
            **clf_kwargs
        )
        clf.fit(X[:, :-1], t_centered, sample_weight=point_weights)

    T, T_std = clf.predict(
        X_predict[:, :-1],
        return_std=True
    )

    k = degree if degree is not None else sum(clf.coef_ != 0) + 1
    return {
        'AIC': AICc(clf.scores_[-1], k, T.shape[0]),
        'BIC': BIC(clf.scores_[-1], k, T.shape[0]),
        'clf': clf,
        'k': k,
        'R': R_predict + R_average,
        'T': T + t_average,
        'T_std': T_std,
    }


def get_polynomial_fits(R, t, degree_range=(2, 20), sample_weight=None):
    trial_degrees = range(*degree_range)
    with Pool(len(trial_degrees)) as p:
        args = (
            ((R, t, d, 100, sample_weight) for d in trial_degrees)
            if sample_weight is not None
            else ((R, t, d, 100) for d in trial_degrees)
        )
        res = p.starmap(polynomial_fit, args)
    return res


def find_optimal_polynomial(R, t, degree_range=(2, 15), method='AIC',
                            **kwargs):
    res = get_polynomial_fits(R, t, degree_range=(2, 15), **kwargs)
    best_index = np.argmin([i.get(method, -1) for i in res])
    return res[best_index]


def pitch_angle_from_xy(x, y):
    dy, dx = (y[2:] - y[:-2]), (x[2:] - x[:-2])
    tangentAngle = np.arctan2(dy/dx, 1)
    r, theta = r_theta_from_xy(x, y)
    pA = np.rad2deg(theta[1:-1] + np.pi/2 - tangentAngle)
    correctedPA = (pA + 90) % 180 - 90
    return correctedPA if np.mean(correctedPA) > 0 else -correctedPA


def get_xy_errors(fit_result):
    x_lower, y_lower = xy_from_r_theta(
        fit_result['R'],
        fit_result['T'] - fit_result['T_std'],
    )
    x_upper, y_upper = xy_from_r_theta(
        fit_result['R'],
        fit_result['T'] + fit_result['T_std'],
    )
    return np.stack((x_lower, y_lower)), np.stack((x_upper, y_upper))


def weighted_group_cross_val(pipeline, X, y, cv, groups, weights):
    scores = np.zeros(cv.get_n_splits(X, y, groups=groups))
    for i, (train, test) in enumerate(cv.split(X, y, groups=groups)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        group_weights = weights[train] / weights[train].mean()
        pipeline.fit(X_train, y_train,
                     bayesianridge__sample_weight=group_weights)
        # scores[i] = pipeline.score(
        #     X_test,
        #     y_test,
        #     sample_weight=weights[test]
        # )
        y_pred = pipeline.predict(
            X_test,
        )
        scores[i] = -median_absolute_error(y_pred, y_test)
    return scores


# pipeline to fit r = a * exp(b * theta)
def get_log_spiral_pipeline():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(
            degree=1,
            include_bias=False,
        ),
        TransformedTargetRegressor(
            regressor=BayesianRidge(
                compute_score=True,
                fit_intercept=True,
                copy_X=True,
                normalize=True,
                **clf_kwargs
            ),
            func=np.log,
            inverse_func=np.exp
        )
    )


# pipeline to fit y = \sum_{i=1}^{degree} c_i * x^i
def get_polynomial_pipeline(degree):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(
            degree=degree,
            include_bias=False,
        ),
        BayesianRidge(
            compute_score=True,
            fit_intercept=True,
            copy_X=True,
            normalize=True,
            **clf_kwargs
        )
    )


def fit_to_points(R, t, **kwargs):
    # perform the log spiral fit
    log_spiral_result = log_spiral_fit(R, t, **kwargs)

    # perform the poly spiral fit
    polynomial_result = find_optimal_polynomial(R, t, **kwargs)

    # get the xy coordinates of the log spiral (and errors)
    log_spiral_xy = np.stack(xy_from_r_theta(
        log_spiral_result['R'],
        log_spiral_result['T'],
    ))
    log_spiral_xy_lower, log_spiral_xy_upper = get_xy_errors(log_spiral_result)

    # get the xy coordinates of the polynomial spiral (and errors)
    polynomial_xy = np.stack(xy_from_r_theta(
        polynomial_result['R'],
        polynomial_result['T']
    ))
    polynomial_xy_lower, polynomial_xy_upper = get_xy_errors(polynomial_result)

    # get the pitch angle of the log spiral from the fit coefficients
    # TODO: this is no longer correct
    log_spiral_pa = 90 - np.rad2deg(
        np.arctan2(1, np.abs(log_spiral_result['clf'].coef_[0]))
    )
    log_spiral_sigma = np.rad2deg(
        np.sqrt(log_spiral_result['clf'].sigma_[0, 0])
    )

    # get the (verying) pitch angle of the polynomial spiral
    pitch_angle_poly = pitch_angle_from_xy(*polynomial_xy)
    pitch_angle_lower_poly = pitch_angle_from_xy(*polynomial_xy_lower)
    pitch_angle_upper_poly = pitch_angle_from_xy(*polynomial_xy_upper)

    return {
        'params': {
            'r_ordered': R,
            'theta_ordered': t,
        },
        'radial': {
            'log_spiral': {
                'T': log_spiral_result['T'],
                'R': log_spiral_result['R'],
                'error': [
                    log_spiral_result['T'] - log_spiral_result['T_std'],
                    log_spiral_result['T'] + log_spiral_result['T_std'],
                ]
            },
            'polynomial': {
                'T': polynomial_result['T'],
                'R': polynomial_result['R'],
                'error': [
                    polynomial_result['T'] - polynomial_result['T_std'],
                    polynomial_result['T'] + polynomial_result['T_std'],
                ]
            },
        },
        'xy_fit': {
            'log_spiral': log_spiral_xy.T,
            'log_spiral_error': [
                log_spiral_xy_lower.T,
                log_spiral_xy_upper.T,
            ],
            'polynomial': polynomial_xy.T,
            'poly_error': [
                polynomial_xy_lower.T,
                polynomial_xy_upper.T
            ],
        },
        'pitch_angle': {
            'log_spiral': [log_spiral_pa, log_spiral_sigma],
            'poly': [
                pitch_angle_poly,
                pitch_angle_lower_poly,
                pitch_angle_upper_poly
            ]
        },
        'clf': {
            'log_spiral': log_spiral_result['clf'],
            'polynomial': polynomial_result['clf'],
            'polynomial-degree': polynomial_result['k'],
        }
    }
