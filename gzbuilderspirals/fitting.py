import numpy as np
from sklearn.linear_model import BayesianRidge
from scipy.interpolate import interp1d
from . import rThetaFromXY, xyFromRTheta


def unwrapAndSortTheta(r, theta):
    unwrapped_theta = np.unwrap(theta)
    point_order_by_theta = np.argsort(unwrapped_theta)
    theta_ordered = unwrapped_theta[point_order_by_theta] + 2 * np.pi
    r_ordered = r[point_order_by_theta]
    return r_ordered, theta_ordered


def logSpiralFit(r, theta):
    br_clf = BayesianRidge(normalize=True, compute_score=True)

    degree = 1
    X_Theta = np.vander(theta, degree + 1)
    br_clf.fit(X_Theta, np.log(r))
    r_br_log_spiral_logged, r_br_log_spiral_std = br_clf.predict(
        X_Theta,
        return_std=True
    )

    return {
        'clf': br_clf,
        'theta': theta,
        'r': np.exp(r_br_log_spiral_logged),
        'errors': (
            np.exp(r_br_log_spiral_logged - 1 * r_br_log_spiral_std),
            np.exp(r_br_log_spiral_logged + 1 * r_br_log_spiral_std)
        )
    }


def splineSpiralFit(r, theta, theta_predict=False, degree=1):
    if theta_predict is False:
        theta_predict = theta
    br_clf = BayesianRidge(normalize=True, compute_score=True)

    X = np.vander(theta, degree+1)
    X_predict = np.vander(theta_predict, degree + 1)
    br_clf.fit(X, r)
    r_br, r_br_std = br_clf.predict(
        X_predict,
        return_std=True
    )

    return {
        'clf': br_clf,
        'theta': theta,
        'r': r_br,
        'errors': (r_br - 1 * r_br_std, r_br + 1 * r_br_std)
    }


def pitchAngleFromXY(x, y):
    dy, dx = (y[2:] - y[:-2]), (x[2:] - x[:-2])
    tangentAngle = np.arctan2(dy/dx, 1)
    r, theta = rThetaFromXY(x, y)
    pA = np.rad2deg(theta[1:-1] + np.pi/2 - tangentAngle)
    correctedPA = (pA + 90) % 180 - 90
    return correctedPA if np.mean(correctedPA) > 0 else -correctedPA


def fitToPoints(arm, spline_degree=3):
    # get the sorting line from the arm
    sortingLine = arm.getSortingLine()
    T_predict = np.linspace(0, 1, sortingLine.shape[0])

    # order the arm along the deprojected XY fit
    o = arm.orderAlongPolyLine(
        sortingLine
    )

    # get the point cloud ordered along the sorting line
    orderedDeprojectedCloud = (
        arm.cleanedCloud[o['pointOrder']]
    )

    # transform to polar coordinates
    r_points, theta_points = rThetaFromXY(
        *arm.normalise(orderedDeprojectedCloud).T,
        mux=0, muy=0
    )

    r_sorting, theta_sorting = rThetaFromXY(
        *(arm.normalise(sortingLine).T)
    )

    # get a function mapping t along the sorting line to the theta values of
    # the line's points
    thetaFunc = interp1d(T_predict, theta_sorting)

    # unwrap theta values and order by them
    r_ordered, theta_ordered = unwrapAndSortTheta(r_points, theta_points)

    # perform the log spiral fit
    log_spiral_res = logSpiralFit(r_ordered, theta_ordered)

    # get the points
    spline_theta = thetaFunc(T_predict)
    spline_theta = np.linspace(
        np.min(theta_ordered),
        np.max(theta_ordered),
        500
    )

    # perform the spline spiral fit
    spline_spiral_res = splineSpiralFit(
        r_ordered,
        theta_ordered,
        theta_predict=spline_theta,
        degree=spline_degree,
    )

    # get the xy coordinates of the log spiral (and errors)
    x_log_spiral, y_log_spiral = xyFromRTheta(
        log_spiral_res['r'],
        theta_ordered
    )
    x_lower_log_spiral, y_lower_log_spiral = xyFromRTheta(
        log_spiral_res['errors'][0],
        theta_ordered
    )
    x_upper_log_spiral, y_upper_log_spiral = xyFromRTheta(
        log_spiral_res['errors'][1],
        theta_ordered
    )

    # get the xy coordinates of the spline spiral (and errors)
    x_spline, y_spline = xyFromRTheta(spline_spiral_res['r'], spline_theta)
    x_lower_spline, y_lower_spline = xyFromRTheta(
        spline_spiral_res['errors'][0],
        spline_theta
    )
    x_upper_spline, y_upper_spline = xyFromRTheta(
        spline_spiral_res['errors'][1],
        spline_theta
    )

    # get the pitch angle of the log spiral from the fit coefficients
    log_spiral_pa = 90 - np.rad2deg(
        np.arctan2(1, np.abs(log_spiral_res['clf'].coef_[0]))
    )
    log_spiral_sigma = np.rad2deg(np.sqrt(log_spiral_res['clf'].sigma_[0, 0]))

    # get the (verying) pitch angle of the spline spiral
    pitch_angle_spline = pitchAngleFromXY(x_spline, y_spline)
    pitch_angle_lower_spline = pitchAngleFromXY(x_lower_spline, y_lower_spline)
    pitch_angle_upper_spline = pitchAngleFromXY(x_upper_spline, y_upper_spline)

    return {
        'radial': {
            'log_spiral': {
                'theta': theta_ordered,
                'r': log_spiral_res['r'],
                'error': log_spiral_res['errors']
            },
            'spline': {
                'theta': spline_theta,
                'r': spline_spiral_res['r'],
                'error': spline_spiral_res['errors'],
            },
        },
        'xy_fit': {
            'log_spiral': np.stack((x_log_spiral, y_log_spiral), axis=1),
            'log_spiral_error': [
                np.stack((x_lower_log_spiral, y_lower_log_spiral), axis=1),
                np.stack((x_upper_log_spiral, y_upper_log_spiral), axis=1),
            ],
            'spline': np.stack((x_spline, y_spline), axis=1),
            'spline_error': [
                np.stack((x_lower_spline, y_lower_spline), axis=1),
                np.stack((x_upper_spline, y_upper_spline), axis=1),
            ],
        },
        'pitch_angle': {
            'log_spiral': [log_spiral_pa, log_spiral_sigma],
            'spline': [
                pitch_angle_spline,
                pitch_angle_lower_spline,
                pitch_angle_upper_spline
            ]
        },
        'clf': {
            'log_spiral': log_spiral_res['clf'],
            'spline': spline_spiral_res['clf'],
            'spline_degree': spline_degree,
        }
    }
