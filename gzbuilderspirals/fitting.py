import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
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


# pipeline to fit Y = a * exp(b * X), don't bother with bias as logsp is self-
# similar
def get_log_spiral_pipeline():
    names = ('standardscaler', 'polynomialfeatures', 'bayesianridge')
    steps = [
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
    ]
    return Pipeline(memory=None, steps=list(zip(names, steps)))


# pipeline to fit y = \sum_{i=1}^{degree} c_i * X^i
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
