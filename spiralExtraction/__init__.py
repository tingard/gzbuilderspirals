import numpy as np
from shapely.geometry import LineString


def rThetaFromXY(x, y, mux=0, muy=0):
    return (
        np.sqrt((x - mux)**2 + (y - muy)**2),
        np.arctan2((y - muy), (x - mux))
    )


def xyFromRTheta(r, theta, mux=0, muy=0):
    return mux + r * np.cos(theta), muy + r * np.sin(theta)


wrapColor = lambda color, s: '{}{}\033[0m'.format(color, s)
red = lambda s: wrapColor('\033[31m', s)
green = lambda s: wrapColor('\033[32m', s)
yellow = lambda s: wrapColor('\033[33m', s)
blue = lambda s: wrapColor('\033[34m', s)
purple = lambda s: wrapColor('\033[35m', s)


def log(s, flag=True):
    if flag:
        print(s)


def getDrawnArms(id, classifications):
    annotationsForSubject = [
        eval(foo) for foo in
        classifications[classifications['subject_ids'] == id]['annotations']
    ]
    try:
        annotationsWithSpiral = [
            c[3]['value'][0]['value']
            for c in annotationsForSubject
            if len(c) > 3 and len(c[3]['value'][0]['value'])
        ]
    except IndexError as e:
        print('{} raised {}'.format(id, e))
        assert False
    spirals = [[a['points'] for a in c] for c in annotationsWithSpiral]
    spiralsWithLengthCut = [
        [[[p['x'], p['y']] for p in a] for a in c]
        for c in spirals if all([len(a) > 5 for a in c])
    ]
    drawnArms = np.array([
        np.array(arm) for classification in spiralsWithLengthCut
        for arm in classification
        if LineString(arm).is_simple
    ])
    return drawnArms


def deprojectArm(phi, ba, arm):
    p = np.deg2rad(phi)
    Xs = (1 / ba) * (arm[:, 0] * np.cos(p) - arm[:, 1] * np.sin(p))
    Ys = 1 * (arm[:, 0] * np.sin(p) + arm[:, 1] * np.cos(p))

    return np.stack((Xs, Ys), axis=1)
