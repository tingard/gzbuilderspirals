import numpy as np
from shapely.geometry import LineString
from scipy.interpolate import splprep, splev, interp1d
import json


def r_theta_from_xy(x, y, mux=0, muy=0):
    return (
        np.sqrt((x - mux)**2 + (y - muy)**2),
        np.arctan2((y - muy), (x - mux))
    )


def xy_from_r_theta(r, theta, mux=0, muy=0):
    return mux + r * np.cos(theta), muy + r * np.sin(theta)


def log(s, flag=True):
    if flag:
        print(s)


def _vprint(v, *args):
    if v:
        print(*args)


null = None
true = True
false = False


def get_drawn_arms(id, classifications, clean=True):
    # classifications must be a pandas DataFrame from zoo csv export
    annotations_for_subject = map(
        json.loads,
        classifications['annotations'][classifications['subject_ids'] == id].values
    )
    spirals = (
        [a['points'] for a in c[3]['value'][0]['value']]
        for c in annotations_for_subject
        if len(c) > 3 and len(c[3]['value'][0]['value'])
    )
    spirals_with_length_cut = (
        [[[p['x'], p['y']] for p in a] for a in c]
        for c in spirals if all([len(a) > 5 for a in c])
    )
    drawn_arms = np.array([
        np.array(arm) for classification in spirals_with_length_cut
        for arm in classification
        if not clean or LineString(arm).is_simple
    ])
    # annotations_for_subject = [
    #     json.loads(i) for i in
    #     classifications['annotations'][classifications['subject_ids'] == id]
    # ]
    # try:
    #     annotations_with_spiral = [
    #         c[3]['value'][0]['value']
    #         for c in annotations_for_subject
    #         if len(c) > 3 and len(c[3]['value'][0]['value'])
    #     ]
    # except IndexError as e:
    #     print('{} raised {}'.format(id, e))
    #     raise(e)
    # spirals = [[a['points'] for a in c] for c in annotations_with_spiral]
    # spirals_with_length_cut = [
    #     [[[p['x'], p['y']] for p in a] for a in c]
    #     for c in spirals if all([len(a) > 5 for a in c])
    # ]
    # drawn_arms = np.array([
    #     np.array(arm) for classification in spirals_with_length_cut
    #     for arm in classification
    #     if not clean or LineString(arm).is_simple
    # ])
    return drawn_arms


def equalize_arm_length(arms, method=np.max):
    new_you = np.linspace(0, 1, method([len(i) for i in arms]))
    return [
        np.array(splev(new_you, splprep(arm.T, s=0, k=1)[0])).T
        for arm in arms
    ]


def weight_r_by_n_arms(R, groups):
    radiuses = [R[groups == g] for g in np.unique(groups)]

    r_bins = np.linspace(np.min(R), np.max(R), 100)
    counts = np.zeros(r_bins.shape)
    for i, _ in enumerate(r_bins[:-1]):
        n = sum(
            1
            for r in radiuses
            if np.any(r >= r_bins[i]) and np.any(r <= r_bins[i+1])
        )
        counts[i] = max(0, n)
    return interp1d(r_bins, counts/sum(counts))


def get_sample_weight(R, groups, bar_length=0):
    w = np.ones(R.shape[0])
    w *= R**2
    w *= weight_r_by_n_arms(R, groups)(R)
    w[R < bar_length] = 0
    w /= np.average(w)
    return w
